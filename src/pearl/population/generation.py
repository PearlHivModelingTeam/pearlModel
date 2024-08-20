"""
Module for the generation of initial population.
"""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pearl.interpolate import restricted_cubic_spline_var
from pearl.parameters import Parameters
from pearl.population.events import calculate_cd4_increase
from pearl.sample import draw_from_trunc_norm


def simulate_ages(
    coeffs: pd.DataFrame, pop_size: int, random_state: np.random.RandomState
) -> NDArray[Any]:
    """Return numpy array of ages with length pop_size drawn from a mixed gaussian of given
    coefficients truncated at 18 and 85.
    """
    # Draw population size of each normal from the binomial distribution
    pop_size_1 = random_state.binomial(pop_size, coeffs.loc["lambda1", "estimate"])
    pop_size_2 = pop_size - pop_size_1

    # Draw ages from truncated normal
    ages_1 = draw_from_trunc_norm(
        18,
        85,
        coeffs.loc["mu1", "estimate"],
        coeffs.loc["sigma1", "estimate"],
        pop_size_1,
        random_state,
    )
    ages_2 = draw_from_trunc_norm(
        18,
        85,
        coeffs.loc["mu2", "estimate"],
        coeffs.loc["sigma2", "estimate"],
        pop_size_2,
        random_state,
    )
    ages = np.concatenate((ages_1, ages_2))
    assert ages.min() > 18
    assert ages.max() < 85
    return np.array(ages)


def simulate_new_dx(
    new_dx: pd.DataFrame,
    linkage_to_care: pd.DataFrame,
    random_state: np.random.RandomState,
) -> Tuple[int, pd.DataFrame]:
    """Return the number of ART non-users in 2009 as an integer and the number of agents entering
    the model each year as art users and non-users as a dataframe. Draw number of new diagnoses
    from a uniform distribution between upper and lower bounds. Calculate number of new art
    initiators by assuming a certain number link in the first year as estimated by a linear
    regression on CDC data, capped at 95%. We assume that 40% of the remaining population links to
    care over the next 3 years. We assume that 70% of those linking to care begin ART, rising
    to 85% in 2011 and 97% afterwards. We take the number of people not initiating ART 2006 - 2009
    in this calculation to be the out of care population size in 2009 for our simulation.
    """

    # Draw new dx from a uniform distribution between upper and lower for 2016-final_year
    new_dx["n_dx"] = new_dx["lower"] + (new_dx["upper"] - new_dx["lower"]) * random_state.uniform()

    # Only a proportion of new diagnoses link to care and 40% of the remaining link
    # in the next 3 years
    new_dx["unlinked"] = new_dx["n_dx"] * (1 - linkage_to_care["link_prob"])
    new_dx["gardner_per_year"] = new_dx["unlinked"] * 0.4 / 3.0
    new_dx["year0"] = new_dx["n_dx"] * linkage_to_care["link_prob"]
    new_dx["year1"] = new_dx["gardner_per_year"].shift(1, fill_value=0)
    new_dx["year2"] = new_dx["gardner_per_year"].shift(2, fill_value=0)
    new_dx["year3"] = new_dx["gardner_per_year"].shift(3, fill_value=0)
    new_dx["total_linked"] = new_dx["year0"] + new_dx["year1"] + new_dx["year2"] + new_dx["year3"]

    # Proportion of those linked to care start ART
    new_dx["art_initiators"] = (new_dx["total_linked"] * linkage_to_care["art_prob"]).astype(int)
    new_dx["art_delayed"] = (new_dx["total_linked"] * (1 - linkage_to_care["art_prob"])).astype(
        int
    )

    # TODO make the start and end dates here parametric
    # Count those not starting art 2006 - 2009 as initial ART nonusers
    n_initial_nonusers = new_dx.loc[np.arange(2006, 2010), "art_delayed"].sum()

    # Compile list of number of new agents to be introduced in the model
    new_agents = new_dx.loc[
        np.arange(2010, new_dx.index.max() + 1), ["art_initiators", "art_delayed"]
    ]

    return n_initial_nonusers, new_agents


def apply_bmi_intervention(
    pop: pd.DataFrame, parameters: Parameters, random_state: np.random.RandomState
) -> pd.DataFrame:
    if parameters.bmi_intervention == 0:
        raise ValueError("Running apply_bmi_intervention despite bmi_intervention=0")
    pop["bmiInt_scenario"] = np.array(parameters.bmi_intervention_scenario, dtype="int8")
    pop["bmiInt_year"] = pop["h1yy"].isin(
        range(
            parameters.bmi_intervention_start_year,
            parameters.bmi_intervention_end_year + 1,
        )
    )  # this function doesnt capture the last value, so we do +1
    pop["bmiInt_coverage"] = np.array(
        random_state.choice(
            [1, 0],
            size=len(pop),
            replace=True,
            p=[
                parameters.bmi_intervention_coverage,
                1 - parameters.bmi_intervention_coverage,
            ],
        ),
        dtype="bool",
    )
    pop["bmiInt_effectiveness"] = np.array(
        random_state.choice(
            [1, 0],
            size=len(pop),
            replace=True,
            p=[
                parameters.bmi_intervention_effectiveness,
                1 - parameters.bmi_intervention_effectiveness,
            ],
        ),
        dtype="bool",
    )
    # determine eligibility:
    pop["bmiInt_ineligible_dm"] = pop["dm"] == 1
    pop["bmiInt_ineligible_underweight"] = pop["pre_art_bmi"] < 18.5
    pop["bmiInt_ineligible_obese"] = pop["pre_art_bmi"] >= 30
    pop["bmiInt_eligible"] = (
        (pop["pre_art_bmi"] >= 18.5) & (pop["pre_art_bmi"] < 30) & (pop["dm"] == 0)
    )

    # eligible people are enrolled in the intervention:
    pop["bmiInt_received"] = pop["bmiInt_eligible"] & pop["bmiInt_year"] & pop["bmiInt_coverage"]

    # creating new outputs:
    pop["post_art_bmi_without_bmiInt"] = pop["post_art_bmi"]
    pop["bmi_increase_postART"] = pop["post_art_bmi"] / pop["pre_art_bmi"] > 1
    pop["bmi_increase_postART_over5p"] = pop["post_art_bmi"] / pop["pre_art_bmi"] > 1.05
    pop["become_obese_postART"] = pop["post_art_bmi_without_bmiInt"] >= 30

    # Scenario0: no BMI intervention
    pop["bmiInt_impacted"] = False

    # Scenario1: Based on BMI threshold:
    # Anyone gaining weight who pass the threshold of BMI=30 (obesity) will experience benefits
    # from this intervention by retaining their weight at a threshold of 29.9 (below obesity)
    if parameters.bmi_intervention_scenario == 1:
        pop["bmiInt_impacted"] = (
            pop["bmiInt_received"] & pop["become_obese_postART"] & pop["bmiInt_effectiveness"]
        )
        pop.loc[pop["bmiInt_impacted"], "post_art_bmi"] = 29.9

    # Scenario2: Based on % gain in BMI :
    # Anyone experiencing >5% increase in pre-ART BMI will experience benefits from this
    # intervention by retaining their BMI at 1.05 times the starting value. Those surpassing the
    # BMI of 30 (obesity threshold) will retain weights at a threshold of 29.9 (below obesity)
    if parameters.bmi_intervention_scenario == 2:
        pop["bmiInt_impacted"] = (
            pop["bmiInt_received"]
            & pop["bmi_increase_postART_over5p"]
            & pop["bmiInt_effectiveness"]
        )
        pop.loc[pop["bmiInt_impacted"], "post_art_bmi"] = np.minimum(
            29.9, 1.05 * pop.loc[pop["bmiInt_impacted"], "pre_art_bmi"]
        )

    # Scenario3: No BMI gain:
    # Anyone gaining BMI will experience benefits from this intervention by retaining their weight
    # at the level of pre-ART BMI. Those experiencing reductions in their weight are allowed to
    # follow the natural weight loss trajectory.
    if parameters.bmi_intervention_scenario == 3:
        pop["bmiInt_impacted"] = (
            pop["bmiInt_received"] & pop["bmi_increase_postART"] & pop["bmiInt_effectiveness"]
        )
        pop.loc[pop["bmiInt_impacted"], "post_art_bmi"] = pop.loc[
            pop["bmiInt_impacted"], "pre_art_bmi"
        ]

    return pop[
        [
            "bmiInt_scenario",
            "bmiInt_ineligible_dm",
            "bmiInt_ineligible_underweight",
            "bmiInt_ineligible_obese",
            "bmiInt_eligible",
            "bmiInt_received",
            "bmi_increase_postART",
            "bmi_increase_postART_over5p",
            "become_obese_postART",
            "bmiInt_impacted",
            "pre_art_bmi",
            "post_art_bmi_without_bmiInt",
            "post_art_bmi",
        ]
    ]


def calculate_post_art_bmi(
    pop: pd.DataFrame,
    parameters: Parameters,
    random_state: np.random.RandomState,
    intervention: Optional[bool] = False,
) -> NDArray[Any]:
    """Calculate and return post art bmi for a population. Sqrt of post art bmi is modeled as a
    linear function of ART initiation year and age, sqrt of pre art bmi, sqrt initial cd4 count,
    and sqrt of cd4 count 2 years after art initiation all modeled as restricted cubic splines.
    """
    # Copy coefficients and knots to more reasonable variable names
    coeffs = parameters.post_art_bmi
    t_age = parameters.post_art_bmi_age_knots
    t_pre_sqrt = parameters.post_art_bmi_pre_art_bmi_knots
    t_sqrtcd4 = parameters.post_art_bmi_cd4_knots
    t_sqrtcd4_post = parameters.post_art_bmi_cd4_post_knots
    rse = parameters.post_art_bmi_rse

    # Calculate spline variables
    pop["age_"] = restricted_cubic_spline_var(pop["init_age"], t_age, 1)
    pop["age__"] = restricted_cubic_spline_var(pop["init_age"], t_age, 2)
    pop["pre_sqrt"] = pop["pre_art_bmi"] ** 0.5
    pop["pre_sqrt_"] = restricted_cubic_spline_var(pop["pre_sqrt"], t_pre_sqrt, 1)
    pop["pre_sqrt__"] = restricted_cubic_spline_var(pop["pre_sqrt"], t_pre_sqrt, 2)
    pop["sqrtcd4"] = pop["init_sqrtcd4n"]
    pop["sqrtcd4_"] = restricted_cubic_spline_var(pop["sqrtcd4"], t_sqrtcd4, 1)
    pop["sqrtcd4__"] = restricted_cubic_spline_var(pop["sqrtcd4"], t_sqrtcd4, 2)

    # Calculate cd4 count 2 years after art initiation and its spline terms
    pop_future = pop.copy().assign(age=pop["init_age"] + 2)
    pop_future["year"] = pop["h1yy"] + 2
    pop_future["age_cat"] = np.floor(pop_future["age"] / 10)
    pop_future.loc[pop_future["age_cat"] < 2, "age_cat"] = 2
    pop_future.loc[pop_future["age_cat"] > 7, "age_cat"] = 7
    pop["sqrtcd4_post"] = calculate_cd4_increase(pop_future, parameters)

    pop["sqrtcd4_post_"] = restricted_cubic_spline_var(pop["sqrtcd4_post"], t_sqrtcd4_post, 1)
    pop["sqrtcd4_post__"] = restricted_cubic_spline_var(pop["sqrtcd4_post"], t_sqrtcd4_post, 2)

    # Create the population matrix and perform the matrix multiplication
    pop_matrix = pop[
        [
            "init_age",
            "age_",
            "age__",
            "h1yy",
            "intercept",
            "pre_sqrt",
            "pre_sqrt_",
            "pre_sqrt__",
            "sqrtcd4",
            "sqrtcd4_",
            "sqrtcd4__",
            "sqrtcd4_post",
            "sqrtcd4_post_",
            "sqrtcd4_post__",
        ]
    ].to_numpy(dtype=float)
    sqrt_post_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
    sqrt_post_art_bmi = sqrt_post_art_bmi.T[0]
    if intervention:
        sqrt_post_art_bmi = np.vectorize(draw_from_trunc_norm)(
            np.sqrt(10), np.sqrt(30), sqrt_post_art_bmi, np.sqrt(rse), 1, random_state
        )
    else:
        sqrt_post_art_bmi = np.vectorize(draw_from_trunc_norm)(
            np.sqrt(10), np.sqrt(65), sqrt_post_art_bmi, np.sqrt(rse), 1, random_state
        )
    post_art_bmi = sqrt_post_art_bmi**2.0

    return np.array(post_art_bmi)


def calculate_pre_art_bmi(
    pop: pd.DataFrame, parameters: Parameters, random_state: np.random.RandomState
) -> NDArray[Any]:
    """Calculate and return pre art bmi for a given population dataframe.
    Each subpopulation can use a different model of pre art bmi.
    """
    # Calculate pre art bmi using one of 5 different models depending on subpopulation
    # Copy coefficients and knots to more reasonable variable names
    coeffs = parameters.pre_art_bmi
    t_age = parameters.pre_art_bmi_age_knots
    t_h1yy = parameters.pre_art_bmi_h1yy_knots
    rse = parameters.pre_art_bmi_rse
    pre_art_bmi = np.nan
    model = parameters.pre_art_bmi_model
    if model == 6:
        pop["age_"] = restricted_cubic_spline_var(pop["init_age"], t_age, 1)
        pop["age__"] = restricted_cubic_spline_var(pop["init_age"], t_age, 2)
        h1yy = pop["h1yy"].values
        pop["h1yy_"] = restricted_cubic_spline_var(h1yy, t_h1yy, 1)
        pop["h1yy__"] = restricted_cubic_spline_var(h1yy, t_h1yy, 2)
        pop_matrix = pop[
            ["init_age", "age_", "age__", "h1yy", "h1yy_", "h1yy__", "intercept"]
        ].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 5:
        pop["age_"] = restricted_cubic_spline_var(pop["init_age"], t_age, 1)
        pop["age__"] = restricted_cubic_spline_var(pop["init_age"], t_age, 2)
        pop_matrix = pop[["init_age", "age_", "age__", "h1yy", "intercept"]].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 3:
        pop_matrix = pop[["init_age", "h1yy", "intercept"]].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 2:
        pop["age_"] = (pop["init_age"] >= 30) & (pop["init_age"] < 40)
        pop["age__"] = (pop["init_age"] >= 40) & (pop["init_age"] < 50)
        pop["age___"] = (pop["init_age"] >= 50) & (pop["init_age"] < 60)
        pop["age____"] = pop["init_age"] >= 60
        h1yy = pop["h1yy"].values
        pop["h1yy_"] = restricted_cubic_spline_var(h1yy, t_h1yy, 1)
        pop["h1yy__"] = restricted_cubic_spline_var(h1yy, t_h1yy, 2)
        pop_matrix = pop[
            [
                "age_",
                "age__",
                "age___",
                "age____",
                "h1yy",
                "h1yy_",
                "h1yy__",
                "intercept",
            ]
        ].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 1:
        pop["age_"] = (pop["init_age"] >= 30) & (pop["init_age"] < 40)
        pop["age__"] = (pop["init_age"] >= 40) & (pop["init_age"] < 50)
        pop["age___"] = (pop["init_age"] >= 50) & (pop["init_age"] < 60)
        pop["age____"] = pop["init_age"] >= 60
        pop_matrix = pop[["age_", "age__", "age___", "age____", "h1yy", "intercept"]].to_numpy(
            dtype=float
        )
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    log_pre_art_bmi = log_pre_art_bmi.T[0]
    log_pre_art_bmi = np.vectorize(draw_from_trunc_norm)(
        np.log10(10), np.log10(65), log_pre_art_bmi, np.sqrt(rse), 1, random_state
    )
    pre_art_bmi = 10.0**log_pre_art_bmi
    return np.array(pre_art_bmi)
