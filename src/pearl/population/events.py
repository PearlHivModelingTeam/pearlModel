from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pearl.definitions import SMEARING
from pearl.interpolate import restricted_cubic_spline_var, restricted_quadratic_spline_var
from pearl.parameters import Parameters


def create_mortality_in_care_pop_matrix(pop: pd.DataFrame, parameters: Parameters) -> NDArray[Any]:
    """
    Return the population matrix as a numpy array for calculating mortality in care. This log odds
    of mortality are a linear function of calendar year, ART init year category modeled as two
    binary variables, and age and sqrt initial cd4 count modeled as restricted cubic splines. If
    using comorbidities, log odds of mortality are a linear function of calendar year, age
    category, initial cd4 count, delta bmi and post art bmi modeled as restricted cubic
    splines, and presence of each individual comorbidity modeled as binary variables.
    """

    pop["post_art_bmi_"] = restricted_cubic_spline_var(
        pop["post_art_bmi"], parameters.mortality_in_care_post_art_bmi, 1
    )
    pop["post_art_bmi__"] = restricted_cubic_spline_var(
        pop["post_art_bmi"], parameters.mortality_in_care_post_art_bmi, 2
    )
    return np.array(
        pop[
            [
                "age_cat",
                "anx",
                "post_art_bmi",
                "post_art_bmi_",
                "post_art_bmi__",
                "ckd",
                "dm",
                "dpr",
                "esld",
                "h1yy",
                "hcv",
                "ht",
                "intercept",
                "lipid",
                "malig",
                "mi",
                "smoking",
                "init_sqrtcd4n",
                "year",
            ]
        ],
        dtype=float,
    )


def create_mortality_out_care_pop_matrix(
    pop: pd.DataFrame, parameters: Parameters
) -> NDArray[Any]:
    """
    Return the population matrix as a numpy array for calculating mortality out of care.
    This log odds of mortality are a linear function of calendar year and age and sqrt cd4 count
    modeled as restricted cubic splines. If using comorbidities, log odds of mortality are a
    linear  function of calendar year, age category, sqrt cd4 count, delta bmi and post art bmi
    modeled as  restricted cubic splines, and presence of each individual comorbidity modeled as
    binary variables.
    """

    pop["post_art_bmi_"] = restricted_cubic_spline_var(
        pop["post_art_bmi"], parameters.mortality_out_care_post_art_bmi, 1
    )
    pop["post_art_bmi__"] = restricted_cubic_spline_var(
        pop["post_art_bmi"], parameters.mortality_out_care_post_art_bmi, 2
    )
    return np.array(
        pop[
            [
                "age_cat",
                "anx",
                "post_art_bmi",
                "post_art_bmi_",
                "post_art_bmi__",
                "ckd",
                "dm",
                "dpr",
                "esld",
                "hcv",
                "ht",
                "intercept",
                "lipid",
                "malig",
                "mi",
                "smoking",
                "time_varying_sqrtcd4n",
                "year",
            ]
        ],
        dtype=float,
    )


def calculate_cd4_increase(pop: pd.DataFrame, parameters: Parameters) -> NDArray[Any]:
    """
    Return new cd4 count of the given population as calculated via a linear function of time
    since art initiation modeled as a spline, initial cd4 count category, age category and
    cross terms.
    """
    knots = parameters.cd4_increase_knots
    coeffs = parameters.cd4_increase.to_numpy(dtype=float)

    # Calculate spline variables
    pop["time_from_h1yy"] = pop["year"] - pop["last_h1yy"]
    pop["time_from_h1yy_"] = restricted_quadratic_spline_var(
        pop["time_from_h1yy"], knots.to_numpy(), 1
    )
    pop["time_from_h1yy__"] = restricted_quadratic_spline_var(
        pop["time_from_h1yy"], knots.to_numpy(), 2
    )
    pop["time_from_h1yy___"] = restricted_quadratic_spline_var(
        pop["time_from_h1yy"], knots.to_numpy(), 3
    )

    # Calculate CD4 Category Variables
    pop["cd4_cat_349"] = (
        pop["last_init_sqrtcd4n"].ge(np.sqrt(200.0)) & pop["last_init_sqrtcd4n"].lt(np.sqrt(350.0))
    ).astype(int)
    pop["cd4_cat_499"] = (
        pop["last_init_sqrtcd4n"].ge(np.sqrt(350.0)) & pop["last_init_sqrtcd4n"].lt(np.sqrt(500.0))
    ).astype(int)
    pop["cd4_cat_500"] = pop["last_init_sqrtcd4n"].ge(np.sqrt(500.0)).astype(int)

    # Create cross term variables
    pop["timecd4cat349_"] = pop["time_from_h1yy_"] * pop["cd4_cat_349"]
    pop["timecd4cat499_"] = pop["time_from_h1yy_"] * pop["cd4_cat_499"]
    pop["timecd4cat500_"] = pop["time_from_h1yy_"] * pop["cd4_cat_500"]
    pop["timecd4cat349__"] = pop["time_from_h1yy__"] * pop["cd4_cat_349"]
    pop["timecd4cat499__"] = pop["time_from_h1yy__"] * pop["cd4_cat_499"]
    pop["timecd4cat500__"] = pop["time_from_h1yy__"] * pop["cd4_cat_500"]
    pop["timecd4cat349___"] = pop["time_from_h1yy___"] * pop["cd4_cat_349"]
    pop["timecd4cat499___"] = pop["time_from_h1yy___"] * pop["cd4_cat_499"]
    pop["timecd4cat500___"] = pop["time_from_h1yy___"] * pop["cd4_cat_500"]

    # Create numpy matrix
    pop_matrix = pop[
        [
            "intercept",
            "time_from_h1yy",
            "time_from_h1yy_",
            "time_from_h1yy__",
            "time_from_h1yy___",
            "cd4_cat_349",
            "cd4_cat_499",
            "cd4_cat_500",
            "age_cat",
            "timecd4cat349_",
            "timecd4cat499_",
            "timecd4cat500_",
            "timecd4cat349__",
            "timecd4cat499__",
            "timecd4cat500__",
            "timecd4cat349___",
            "timecd4cat499___",
            "timecd4cat500___",
        ]
    ].to_numpy(dtype=float)

    # Perform matrix multiplication
    new_cd4 = np.matmul(pop_matrix, coeffs)

    new_cd4 = np.clip(new_cd4, 0, np.sqrt(2000))
    return np.array(new_cd4)


def calculate_cd4_decrease(
    pop: pd.DataFrame, parameters: Parameters, smearing: Optional[float] = SMEARING
) -> NDArray[Any]:
    """Calculate out of care cd4 count via a linear function of years out of care and sqrt cd4
    count at exit from care.
    """

    coeffs = parameters.cd4_decrease.to_numpy(dtype=float)

    # Calculate the time_out variable and perform the matrix multiplication
    pop["time_out"] = pop["year"] - pop["ltfu_year"]
    pop_matrix = pop[["intercept", "time_out", "sqrtcd4n_exit"]].to_numpy(dtype=float)
    diff = np.matmul(pop_matrix, coeffs)

    new_cd4 = np.sqrt((pop["sqrtcd4n_exit"].to_numpy(dtype=float) ** 2) * np.exp(diff) * smearing)
    new_cd4 = np.clip(new_cd4, 0, np.sqrt(2000))
    return np.array(new_cd4)
