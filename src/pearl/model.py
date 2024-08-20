# Imports
import os

# TODO move this somewhere better, like into docker
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# TODO refactor this into a single data structure
from pearl.definitions import (
    ART_NAIVE,
    ART_NONUSER,
    ART_USER,
    DEAD_ART_NONUSER,
    DEAD_ART_USER,
    DELAYED,
    DYING_ART_NONUSER,
    DYING_ART_USER,
    LTFU,
    POPULATION_TYPE_DICT,
    REENGAGED,
    STAGE0,
    STAGE1,
    STAGE2,
    STAGE3,
    ALL_COMORBIDITIES
)
from pearl.interpolate import restricted_quadratic_spline_var
from pearl.parameters import Parameters
from pearl.population.events import (
    calculate_cd4_decrease,
    calculate_cd4_increase,
    create_mortality_in_care_pop_matrix,
    create_mortality_out_care_pop_matrix,
)
from pearl.population.generation import (
    apply_bmi_intervention,
    calculate_post_art_bmi,
    calculate_pre_art_bmi,
    simulate_ages,
    simulate_new_dx,
)
from pearl.multimorbidity import create_comorbidity_pop_matrix, create_mm_detail_stats
from pearl.sample import draw_from_trunc_norm

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.options.mode.chained_assignment = None  # default='warn'

###############################################################################
# Pearl Class                                                                 #
###############################################################################

class Pearl:
    """The PEARL class runs a simulation when initialized."""

    def __init__(self, parameters: Parameters, group_name: str, replication: int):
        """
        Takes an instance of the Parameters class, the group name and replication number and 
        runs a simulation.
        """
        self.group_name = group_name
        self.replication = replication
        self.year = 2009
        self.parameters = parameters
        if self.parameters.seed is not None:
            self.random_state = np.random.RandomState(seed=self.parameters.seed)
        else:
            raise ValueError("Parameter should have a seed property")

        with Path.open(self.parameters.output_folder / "random.state", "w") as state_file:
            state_file.write(str(self.parameters.seed))

        # Initiate output class
        self.stats = Statistics(
            output_folder=self.parameters.output_folder,
            group_name=group_name,
            replication=replication,
        )

        # Simulate number of new art initiators and initial nonusers
        n_initial_nonusers, n_new_agents = simulate_new_dx(
            self.parameters.new_dx.copy(),
            self.parameters.linkage_to_care,
            self.random_state,
        )

        # Create art using 2009 population
        user_pop = self.make_user_pop_2009()

        # Create art non-using 2009 population
        non_user_pop = self.make_nonuser_pop_2009(n_initial_nonusers, self.random_state)

        # concat to get initial population
        self.population = pd.concat([user_pop, non_user_pop])

        # Create population of new art initiators
        art_pop = self.make_new_population(n_new_agents, self.random_state)

        # concat the art pop to population
        self.population = (
            pd.concat([self.population, art_pop])
            .fillna(0)
            .astype(
                {
                    "become_obese_postART": "bool",
                    "bmiInt_eligible": "bool",
                    "bmiInt_impacted": "bool",
                    "bmiInt_ineligible_dm": "bool",
                    "bmiInt_ineligible_obese": "bool",
                    "bmiInt_ineligible_underweight": "bool",
                    "bmiInt_received": "bool",
                    "bmiInt_scenario": "int8",
                    "bmi_increase_postART": "bool",
                    "bmi_increase_postART_over5p": "bool",
                }
            )
        )

        # First recording of stats
        self.record_stats()

        # Print populations
        if self.parameters.verbose:
            print(self)

        # Move to 2010
        self.year += 1

        # Run
        self.run()

        # Save output
        self.stats.save()

        self.population = self.population.assign(
            group=self.group_name, replication=self.replication
        )
        self.population.to_parquet(self.parameters.output_folder / "population.parquet")
        
    @staticmethod
    def calculate_prob(pop: pd.DataFrame, coeffs: NDArray[Any]) -> NDArray[Any]:
        """
        Calculate and return a numpy array of individual probabilities from logistic regression 
        given the population and coefficient matrices.
        Used for multiple logistic regression functions.
        """
        # Calculate log odds using a matrix multiplication
        log_odds = np.matmul(pop, coeffs)

        # Convert to probability
        prob = np.exp(log_odds) / (1.0 + np.exp(log_odds))
        return np.array(prob)
    
    @staticmethod
    def create_ltfu_pop_matrix(pop: pd.DataFrame, knots: pd.DataFrame) -> NDArray[Any]:
        """
        Create and return the population matrix as a numpy array for use in calculating probability 
        of loss to follow up.
        """
        # Create all needed intermediate variables
        pop["age_"] = restricted_quadratic_spline_var(pop["age"], knots.to_numpy(), 1)
        pop["age__"] = restricted_quadratic_spline_var(pop["age"], knots.to_numpy(), 2)
        pop["age___"] = restricted_quadratic_spline_var(pop["age"], knots.to_numpy(), 3)
        pop["haart_period"] = (pop["h1yy"].values > 2010).astype(int)
        return np.array(
            pop[
                [
                    "intercept",
                    "age",
                    "age_",
                    "age__",
                    "age___",
                    "year",
                    "init_sqrtcd4n",
                    "haart_period",
                ]
            ]
        )

    def make_user_pop_2009(self) -> pd.DataFrame:
        """Create and return initial 2009 population dataframe. Draw ages from a mixed normal 
        distribution truncated at 18 and 85. Assign ART initiation year using proportions from 
        NA-ACCORD data. Draw sqrt CD4 count from a normal distribution truncated at 0 and 
        sqrt(2000). If doing an Aim 2 simulation, assign bmi, comorbidities, and multimorbidity
        using their respective models.
        """
        # Create population dataframe
        population = pd.DataFrame()

        # Draw ages from the truncated mixed gaussian
        n_initial_users = self.parameters.on_art_2009.iloc[0]
        population["age"] = simulate_ages(
            self.parameters.age_in_2009, n_initial_users, self.random_state
        )
        population.loc[population["age"] < 18, "age"] = 18
        population.loc[population["age"] > 85, "age"] = 85

        # Create age categories
        population["age"] = np.floor(population["age"])
        population["age_cat"] = np.floor(population["age"] / 10)
        population.loc[population["age_cat"] > 7, "age_cat"] = 7
        population["id"] = np.array(range(population.index.size))
        population = population.sort_values("age")
        population = population.set_index(["age_cat", "id"])

        # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
        for age_cat, grouped in population.groupby("age_cat"):
            h1yy_data = self.parameters.h1yy_by_age_2009.loc[age_cat].reset_index()
            population.loc[age_cat, "h1yy"] = self.random_state.choice(
                h1yy_data["h1yy"], size=len(grouped), p=h1yy_data["pct"]
            )

        # Reindex for group operation
        population["h1yy"] = population["h1yy"].astype(int)
        population = population.reset_index().set_index(["h1yy", "id"]).sort_index()

        # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 and sqrt 2000
        for h1yy, group in population.groupby(level=0):
            mu = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, "mu"), "estimate"]
            sigma = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, "sigma"), "estimate"]
            size = group.shape[0]
            sqrt_cd4n = draw_from_trunc_norm(
                0, np.sqrt(2000.0), mu, sigma, size, self.random_state
            )
            population.loc[(h1yy,), "init_sqrtcd4n"] = sqrt_cd4n
        population = population.reset_index().set_index("id").sort_index()

        # Toss out age_cat < 2
        population.loc[population["age_cat"] < 2, "age_cat"] = 2

        # Add final columns used for calculations and output
        population["last_h1yy"] = population["h1yy"]
        population["last_init_sqrtcd4n"] = population["init_sqrtcd4n"]
        population["init_age"] = population["age"] - (2009 - population["h1yy"])
        population["n_lost"] = np.array(0, dtype="int32")
        population["years_out"] = np.array(0, dtype="int16")
        population["year_died"] = np.nan
        population["sqrtcd4n_exit"] = 0
        population["ltfu_year"] = np.array(0, dtype="int16")
        population["return_year"] = np.array(0, dtype="int16")
        population["intercept"] = 1.0
        population["year"] = np.array(2009, dtype="int16")

        # Calculate time varying cd4 count
        population["time_varying_sqrtcd4n"] = calculate_cd4_increase(
            population.copy(), self.parameters
        )

        # Set status and initiate out of care variables
        population["status"] = ART_USER

        # add bmi, comorbidity, and multimorbidity columns

        # Bmi
        population["pre_art_bmi"] = calculate_pre_art_bmi(
            population.copy(), self.parameters, self.random_state
        )
        population["post_art_bmi"] = calculate_post_art_bmi(
            population.copy(), self.parameters, self.random_state
        )
        population["delta_bmi"] = population["post_art_bmi"] - population["pre_art_bmi"]

        # Apply comorbidities
        for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
            population[condition] = (
                self.random_state.rand(len(population.index))
                < self.parameters.prev_users_dict[condition].values
            ).astype(int)
            population[f"t_{condition}"] = np.array(0, dtype="int8")
        population["mm"] = np.array(population[STAGE2 + STAGE3].sum(axis=1), dtype="int8")

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        population = population.astype(POPULATION_TYPE_DICT)

        return population

    def make_nonuser_pop_2009(
        self, n_initial_nonusers: pd.DataFrame, random_state: np.random.RandomState
    ) -> pd.DataFrame:
        """Create and return initial 2009 population dataframe. Draw ages from a mixed normal 
        distribution truncated at 18 and 85. Assign ART initiation year using proportions from 
        NA-ACCORD data. Draw sqrt CD4 count from a normal distribution truncated at 0 and 
        sqrt(2000). If doing an Aim 2 simulation, assign bmi, comorbidities, and multimorbidity
        using their respective models.
        """
        # Create population dataframe
        population = pd.DataFrame()

        # Draw ages from the truncated mixed gaussian
        population["age"] = simulate_ages(
            self.parameters.age_in_2009, n_initial_nonusers, random_state
        )
        population.loc[population["age"] < 18, "age"] = 18
        population.loc[population["age"] > 85, "age"] = 85

        # Create age categories
        population["age"] = np.floor(population["age"])
        population["age_cat"] = np.floor(population["age"] / 10)
        population.loc[population["age_cat"] > 7, "age_cat"] = 7
        population["id"] = range(population.index.size)
        population = population.sort_values("age")
        population = population.set_index(["age_cat", "id"])

        # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
        for age_cat, grouped in population.groupby("age_cat"):
            h1yy_data = self.parameters.h1yy_by_age_2009.loc[age_cat].reset_index()
            population.loc[age_cat, "h1yy"] = random_state.choice(
                h1yy_data["h1yy"], size=len(grouped), p=h1yy_data["pct"]
            )

        # Reindex for group operation
        population["h1yy"] = population["h1yy"].astype(int)
        population = population.reset_index().set_index(["h1yy", "id"]).sort_index()

        # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 and sqrt 2000
        for h1yy, group in population.groupby(level=0):
            mu = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, "mu"), "estimate"]
            sigma = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, "sigma"), "estimate"]
            size = group.shape[0]
            sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size, random_state)
            population.loc[(h1yy,), "init_sqrtcd4n"] = sqrt_cd4n
        population = population.reset_index().set_index("id").sort_index()

        # Toss out age_cat < 2
        population.loc[population["age_cat"] < 2, "age_cat"] = 2

        # Add final columns used for calculations and output
        population["last_h1yy"] = population["h1yy"]
        population["last_init_sqrtcd4n"] = population["init_sqrtcd4n"]
        population["init_age"] = population["age"] - (2009 - population["h1yy"])
        population["n_lost"] = 0
        population["years_out"] = 0
        population["year_died"] = np.nan
        population["sqrtcd4n_exit"] = 0
        population["ltfu_year"] = 0
        population["return_year"] = 0
        population["intercept"] = 1.0
        population["year"] = 2009

        # Calculate time varying cd4 count
        population["time_varying_sqrtcd4n"] = calculate_cd4_increase(
            population.copy(), self.parameters
        )

        # Set status and initiate out of care variables
        years_out_of_care = random_state.choice(
            a=self.parameters.years_out_of_care["years"],
            size=n_initial_nonusers,
            p=self.parameters.years_out_of_care["probability"],
        )
        population["status"] = ART_NONUSER
        population["sqrtcd4n_exit"] = population["time_varying_sqrtcd4n"]
        population["ltfu_year"] = 2009
        population["return_year"] = 2009 + years_out_of_care
        population["n_lost"] += 1

        # add bmi, comorbidity, and multimorbidity columns

        # Bmi
        population["pre_art_bmi"] = calculate_pre_art_bmi(
            population.copy(), self.parameters, random_state
        )
        population["post_art_bmi"] = calculate_post_art_bmi(
            population.copy(), self.parameters, random_state
        )
        population["delta_bmi"] = population["post_art_bmi"] - population["pre_art_bmi"]

        # Apply comorbidities
        for condition in STAGE0:
            population[condition] = (
                random_state.rand(len(population.index))
                < self.parameters.prev_users_dict[condition].values
            ).astype(int)
            population[f"t_{condition}"] = population[
                condition
            ]  # 0 if not having a condition, and 1 if they have it
        for condition in STAGE1 + STAGE2 + STAGE3:
            population[condition] = (
                random_state.rand(len(population.index))
                < (self.parameters.prev_users_dict[condition].values)
            ).astype(int)
            population[f"t_{condition}"] = population[
                condition
            ]  # 0 if not having a condition, and 1 if they have it
        population["mm"] = population[STAGE2 + STAGE3].sum(axis=1)

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        population = population.astype(POPULATION_TYPE_DICT)

        return population

    def make_new_population(
        self, n_new_agents: pd.DataFrame, random_state: np.random.RandomState
    ) -> pd.DataFrame:
        """Create and return the population initiating ART during the simulation. Age and CD4
        count distribution parameters are taken from a linear regression until 2018 and drawn from 
        a uniform distribution between the 2018 values and the predicted values thereafter. Ages
        are drawn from the two-component mixed normal distribution truncated at 18 and 85 defined 
        by the generated parameters. The n_new_agents dataframe defines the population size of ART 
        initiators and those not initiating immediately. The latter population begins ART some 
        years later as drawn from a normalized, truncated Poisson distribution. The sqrt CD4 count 
        at ART initiation for each agent is drawn from a normal distribution truncated at 0 and 
        sqrt 2000 as defined by the generated parameters. If this is an Aim 2 simulation, generate 
        bmi, comorbidities, and multimorbidity from their respective distributions.
        """
        # Draw a random value between predicted and 2018 predicted value for years greater than 
        # 2018
        rand = random_state.rand(len(self.parameters.age_by_h1yy.index))
        self.parameters.age_by_h1yy["estimate"] = (
            rand
            * (
                self.parameters.age_by_h1yy["high_value"]
                - self.parameters.age_by_h1yy["low_value"]
            )
        ) + self.parameters.age_by_h1yy["low_value"]
        self.stats.art_coeffs = (  # type: ignore[attr-defined]
            self.parameters.age_by_h1yy[["estimate"]]
            .assign(variable="age")
            .reset_index()
            .astype({"h1yy": "int16", "param": str, "variable": str})
        )

        rand = random_state.rand(len(self.parameters.cd4n_by_h1yy.index))
        self.parameters.cd4n_by_h1yy["estimate"] = (
            rand
            * (
                self.parameters.cd4n_by_h1yy["high_value"]
                - self.parameters.cd4n_by_h1yy["low_value"]
            )
        ) + self.parameters.cd4n_by_h1yy["low_value"]
        art_coeffs_cd4 = (
            self.parameters.cd4n_by_h1yy[["estimate"]]
            .assign(variable="cd4")
            .reset_index()
            .astype({"h1yy": "int16", "param": str, "variable": str})
        )
        self.stats.art_coeffs = pd.concat([self.stats.art_coeffs, art_coeffs_cd4]).rename(  # type: ignore[attr-defined]
            columns={"h1yy": "year"}
        )[["year", "variable", "param", "estimate"]]

        # Create population
        population = pd.DataFrame()

        # Generate ages and art status for each new initiator based on year of initiation
        for h1yy in self.parameters.age_by_h1yy.index.levels[0]:
            grouped_pop = pd.DataFrame()
            n_initiators = n_new_agents.loc[h1yy, "art_initiators"]
            n_delayed = n_new_agents.loc[h1yy, "art_delayed"]
            grouped_pop["age"] = simulate_ages(
                self.parameters.age_by_h1yy.loc[h1yy],
                n_initiators + n_delayed,
                random_state,
            )
            grouped_pop["h1yy"] = h1yy
            grouped_pop["status"] = ART_NAIVE
            delayed = random_state.choice(a=len(grouped_pop.index), size=n_delayed, replace=False)
            grouped_pop.loc[delayed, "status"] = DELAYED
            population = pd.concat([population, grouped_pop])

        population.loc[population["age"] < 18, "age"] = 18
        population.loc[population["age"] > 85, "age"] = 85

        # Generate number of years for delayed initiators to wait before beginning care and modify 
        # their start year accordingly
        delayed = population["status"] == DELAYED
        years_out_of_care = random_state.choice(
            a=self.parameters.years_out_of_care["years"],
            size=len(population.loc[delayed]),
            p=self.parameters.years_out_of_care["probability"],
        )
        population.loc[delayed, "h1yy"] = population.loc[delayed, "h1yy"] + years_out_of_care
        population.loc[delayed, "status"] = ART_NAIVE
        population = population[population["h1yy"] <= self.parameters.final_year].copy()

        # Create age_cat variable
        population["age"] = np.floor(population["age"])
        population["age_cat"] = np.floor(population["age"] / 10)
        population.loc[population["age_cat"] < 2, "age_cat"] = 2
        population.loc[population["age_cat"] > 7, "age_cat"] = 7

        # Add id number
        population["id"] = np.arange(
            len(self.population), (len(self.population) + population.index.size)
        )

        population.reset_index()
        unique_h1yy = population["h1yy"].unique()
        population["init_sqrtcd4n"] = 0.0
        for h1yy in unique_h1yy:
            mu = self.parameters.cd4n_by_h1yy.loc[(h1yy, "mu"), "estimate"]
            sigma = self.parameters.cd4n_by_h1yy.loc[(h1yy, "sigma"), "estimate"]
            size = len(population[population["h1yy"] == h1yy]["init_sqrtcd4n"])
            sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size, random_state)
            population.loc[population["h1yy"] == h1yy, "init_sqrtcd4n"] = sqrt_cd4n

        population = population.reset_index().set_index("id").sort_index()

        # Calculate time varying cd4 count and other needed variables
        population["last_h1yy"] = population["h1yy"]
        population["time_varying_sqrtcd4n"] = population["init_sqrtcd4n"]
        population["last_init_sqrtcd4n"] = population["init_sqrtcd4n"]
        population["init_age"] = population["age"]
        population["n_lost"] = 0
        population["years_out"] = 0
        population["year_died"] = np.nan
        population["sqrtcd4n_exit"] = 0
        population["ltfu_year"] = 0
        population["return_year"] = 0
        population["intercept"] = 1.0
        population["year"] = 2009

        # Prevalence of existing comorbidities and BMI dynamics:

        # Pre-exisiting comorbidities:
        for condition in STAGE0:
            population[condition] = (
                random_state.rand(len(population.index))
                < self.parameters.prev_inits_dict[condition].values
            ).astype(int)
            population[f"t_{condition}"] = population[
                condition
            ]  # 0 if not having a condition, and 1 if they have it
        for condition in STAGE1 + STAGE2 + STAGE3:
            population[condition] = (
                random_state.rand(len(population.index))
                < (self.parameters.prev_inits_dict[condition].values)
            ).astype(int)
            population[f"t_{condition}"] = population[
                condition
            ]  # 0 if not having a condition, and 1 if they have it
        population["mm"] = population[STAGE2 + STAGE3].sum(axis=1)

        # pre- / post-ART BMI:
        population["pre_art_bmi"] = calculate_pre_art_bmi(
            population.copy(), self.parameters, random_state
        )
        population["post_art_bmi"] = calculate_post_art_bmi(
            population.copy(), self.parameters, random_state
        )

        # Apply post_art_bmi intervention 
        # (eligibility may depend on current exisiting comorbidities)
        if self.parameters.bmi_intervention:
            population[
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
            ] = apply_bmi_intervention(population.copy(), self.parameters, random_state)

        population["delta_bmi"] = population["post_art_bmi"] - population["pre_art_bmi"]

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        # Concat new population to pearl population
        population = population.astype(POPULATION_TYPE_DICT)

        return population

    def run(self) -> None:
        """Simulate from 2010 to final_year"""
        while self.year <= self.parameters.final_year:
            # Increment calendar year, ages, age_cat and years out of care
            self.increment_years()

            # Apply comorbidities
            self.apply_comorbidity_incidence()
            self.update_mm()

            # In care operations
            self.increase_cd4_count()  # Increase cd4n in people in care
            self.add_new_user()  # Add in newly diagnosed ART initiators
            self.kill_in_care()  # Kill some people in care
            self.lose_to_follow_up()  # Lose some people to follow up

            # Out of care operations
            self.decrease_cd4_count()  # Decrease cd4n in people out of care
            self.kill_out_care()  # Kill some people out of care
            self.reengage()  # Reengage some people out of care

            # Record output statistics
            self.record_stats()

            # Append changed populations to their respective DataFrames
            self.append_new()

            # Print population breakdown if verbose
            if self.parameters.verbose:
                print(self)

            # Increment year
            self.year += 1

        # Record output statistics for the end of the simulation
        self.record_final_stats()

    def increment_years(self) -> None:
        """
        Increment calendar year for all agents, increment age and age_cat for those alive in the
        model, and increment the number of years spent out of care for the ART non-using 
        population.
        """
        alive_and_initiated = self.population["status"].isin([ART_USER, ART_NONUSER])
        out_care = self.population["status"] == ART_NONUSER
        self.population["year"] = np.array(self.year, dtype="int16")
        self.population.loc[alive_and_initiated, "age"] += np.array(1, dtype="int8")
        self.population["age_cat"] = np.floor(self.population["age"] / 10).astype("int8")
        self.population.loc[self.population["age_cat"] < 2, "age_cat"] = np.array(2, dtype="int8")
        self.population.loc[self.population["age_cat"] > 7, "age_cat"] = np.array(7, dtype="int8")
        self.population.loc[out_care, "years_out"] += np.array(1, dtype="int8")

    def increase_cd4_count(self) -> None:
        """Calculate and set new CD4 count for ART using population."""
        in_care = self.population["status"] == ART_USER

        new_sqrt_cd4 = calculate_cd4_increase(self.population.loc[in_care].copy(), self.parameters)

        self.population.loc[in_care, "time_varying_sqrtcd4n"] = new_sqrt_cd4

    def decrease_cd4_count(self) -> None:
        """Calculate and set new CD4 count for ART non-using population."""
        out_care = self.population["status"] == ART_NONUSER
        new_sqrt_cd4 = calculate_cd4_decrease(
            self.population.loc[out_care].copy(), self.parameters
        )

        self.population.loc[out_care, "time_varying_sqrtcd4n"] = new_sqrt_cd4

    def add_new_user(self) -> None:
        """Add newly initiating ART users."""
        new_user = (self.population["status"] == ART_NAIVE) & (
            self.population["h1yy"] == self.year
        )
        self.population.loc[new_user, "status"] = ART_USER

    def kill_in_care(self) -> None:
        """Calculate probability of mortality for in care population. Optionally, use the general
        population mortality threshold to increase age category grouped probability of mortality to
        have the same mean as the general population. Draw random numbers to determine
        who will die.
        """
        # Calculate death probability
        in_care = self.population["status"] == ART_USER
        pop = self.population.copy()
        coeff_matrix = self.parameters.mortality_in_care_co.to_numpy(dtype=float)

        pop_matrix = create_mortality_in_care_pop_matrix(pop.copy(), parameters=self.parameters)

        pop["death_prob"] = self.calculate_prob(
            pop_matrix,
            coeff_matrix,
        )

        # Increase mortality to general population threshold
        if self.parameters.mortality_threshold_flag:
            pop["mortality_age_group"] = pd.cut(
                pop["age"],
                bins=[0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85],
                right=True,
                labels=np.arange(14),
            )
            mean_mortality = pd.DataFrame(
                pop.loc[in_care]
                .groupby(["mortality_age_group"], observed=False)["death_prob"]
                .mean()
            )
            mean_mortality["p"] = (
                self.parameters.mortality_threshold["p"] - mean_mortality["death_prob"]
            )
            mean_mortality.loc[mean_mortality["p"] <= 0, "p"] = 0
            for mortality_age_group in np.arange(14):
                excess_mortality = mean_mortality.loc[mortality_age_group, "p"]
                pop.loc[
                    in_care & (pop["mortality_age_group"] == mortality_age_group),
                    "death_prob",
                ] += excess_mortality

        # Draw for mortality
        died = (
            (pop["death_prob"] > self.random_state.rand(len(self.population.index)))
            | (self.population["age"] > 85)
        ) & in_care
        self.population.loc[died, "status"] = DYING_ART_USER
        self.population.loc[died, "year_died"] = np.array(self.year, dtype="int16")

    def kill_out_care(self) -> None:
        """Calculate probability of mortality for out of care population. Optionally, use the
        general population mortality threshold to increase age category grouped probability of
        mortality to have the same mean as the general population. Draw random
        numbers to determine who will die.
        """
        # Calculate death probability
        out_care = self.population["status"] == ART_NONUSER
        pop = self.population.copy()
        coeff_matrix = self.parameters.mortality_out_care_co.to_numpy(dtype=float)

        pop_matrix = create_mortality_out_care_pop_matrix(pop.copy(), parameters=self.parameters)

        pop["death_prob"] = self.calculate_prob(pop_matrix, coeff_matrix)

        # Increase mortality to general population threshold
        if self.parameters.mortality_threshold_flag:
            pop["mortality_age_group"] = pd.cut(
                pop["age"],
                bins=[0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85],
                right=True,
                labels=np.arange(14),
            )
            mean_mortality = pd.DataFrame(
                pop.loc[out_care]
                .groupby(["mortality_age_group"], observed=False)["death_prob"]
                .mean()
            )
            mean_mortality["p"] = (
                self.parameters.mortality_threshold["p"] - mean_mortality["death_prob"]
            )
            mean_mortality.loc[mean_mortality["p"] <= 0, "p"] = 0
            for mortality_age_group in np.arange(14):
                excess_mortality = mean_mortality.loc[mortality_age_group, "p"]
                pop.loc[
                    out_care & (pop["mortality_age_group"] == mortality_age_group),
                    "death_prob",
                ] += excess_mortality

        # Draw for mortality
        died = (
            (pop["death_prob"] > self.random_state.rand(len(self.population.index)))
            | (self.population["age"] > 85)
        ) & out_care
        self.population.loc[died, "status"] = DYING_ART_NONUSER
        self.population.loc[died, "year_died"] = np.array(self.year, dtype="int16")
        self.population.loc[died, "return_year"] = 0

    def lose_to_follow_up(self) -> None:
        """Calculate probability of in care agents leaving care. Draw random number to decide who
        leaves care. For those leaving care, draw the number of years to spend out of care from a
        normalized, truncated Poisson distribution.
        """
        # Calculate probability and draw
        in_care = self.population["status"] == ART_USER
        pop = self.population.copy()
        coeff_matrix = self.parameters.loss_to_follow_up.to_numpy(dtype=float)
        pop_matrix = self.create_ltfu_pop_matrix(pop.copy(), self.parameters.ltfu_knots)
        pop["ltfu_prob"] = self.calculate_prob(
            pop_matrix,
            coeff_matrix,
        )

        lost = (pop["ltfu_prob"] > self.random_state.rand(len(self.population.index))) & in_care

        p = self.parameters.years_out_of_care["probability"]

        years_out_of_care = self.random_state.choice(
            a=self.parameters.years_out_of_care["years"],
            size=len(self.population.loc[lost]),
            p=p,
        )

        # Set variables for lost population
        self.population.loc[lost, "return_year"] = (self.year + years_out_of_care).astype("int16")
        self.population.loc[lost, "status"] = LTFU
        self.population.loc[lost, "sqrtcd4n_exit"] = self.population.loc[
            lost, "time_varying_sqrtcd4n"
        ]
        self.population.loc[lost, "ltfu_year"] = self.year
        self.population.loc[lost, "n_lost"] += 1

    def reengage(self) -> None:
        """Move out of care population scheduled to reenter care."""
        out_care = self.population["status"] == ART_NONUSER
        reengaged = (self.year == self.population["return_year"]) & out_care
        self.population.loc[reengaged, "status"] = REENGAGED

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        self.population.loc[reengaged, "last_init_sqrtcd4n"] = self.population.loc[
            reengaged, "time_varying_sqrtcd4n"
        ]
        self.population.loc[reengaged, "last_h1yy"] = self.year
        self.population.loc[reengaged, "return_year"] = 0

        # Save years out of care
        years_out = (
            pd.DataFrame(self.population.loc[reengaged, "years_out"].value_counts())
            .reindex(range(1, 16), fill_value=0)
            .reset_index()
            .rename(columns={"count": "n"})
            .assign(year=self.year)
        )[["year", "years_out", "n"]].astype({"year": "int16", "years_out": "int8", "n": "int32"})

        self.stats.years_out = pd.concat([self.stats.years_out, years_out])  # type: ignore[attr-defined]
        self.population.loc[reengaged, "years_out"] = 0

    def append_new(self) -> None:
        """Move agents from the temporary, statuses to the main statuses at the end of the year."""
        reengaged = self.population["status"] == REENGAGED
        ltfu = self.population["status"] == LTFU
        dying_art_user = self.population["status"] == DYING_ART_USER
        dying_art_nonuser = self.population["status"] == DYING_ART_NONUSER

        self.population.loc[reengaged, "status"] = ART_USER
        self.population.loc[ltfu, "status"] = ART_NONUSER
        self.population.loc[dying_art_user, "status"] = DEAD_ART_USER
        self.population.loc[dying_art_nonuser, "status"] = DEAD_ART_NONUSER

    def apply_comorbidity_incidence(self) -> None:
        """Calculate probability of incidence of all comorbidities and then draw to determine which
        agents experience incidence. Record incidence data stratified by care status and age
        category.
        """
        in_care = self.population["status"] == ART_USER
        out_care = self.population["status"] == ART_NONUSER

        # Iterate over all comorbidities
        for condition in STAGE1 + STAGE2 + STAGE3:
            # Calculate probability
            coeff_matrix = self.parameters.comorbidity_coeff_dict[condition].to_numpy(dtype=float)
            pop_matrix = create_comorbidity_pop_matrix(
                self.population.copy(), condition=condition, parameters=self.parameters
            )
            prob = self.calculate_prob(
                pop_matrix,
                coeff_matrix,
            )

            # Draw for incidence
            rand = prob > self.random_state.rand(len(self.population.index))
            old = self.population[condition]
            new = rand & (in_care | out_care) & ~old  # new incident comorbidities
            self.population[condition] = (old | new).astype("bool")
            # Update time of incident comorbidity
            self.population[f"t_{condition}"] = np.array(
                self.population[f"t_{condition}"] + new * self.year, dtype="int16"
            )  # we keep the exisiting values

            # Save incidence statistics
            incidence_in_care = (
                self.population.loc[new & in_care]
                .groupby(["age_cat"])
                .size()
                .reindex(index=self.parameters.AGE_CATS, fill_value=0)
                .reset_index(name="n")
                .assign(year=self.year, condition=condition)
            )[["condition", "year", "age_cat", "n"]].astype(
                {"year": "int16", "age_cat": "int8", "n": "int32"}
            )
            self.stats.incidence_in_care = pd.concat(  # type: ignore[attr-defined]
                [self.stats.incidence_in_care, incidence_in_care]
            )

            incidence_out_care = (
                self.population.loc[new & out_care]
                .groupby(["age_cat"])
                .size()
                .reindex(index=self.parameters.AGE_CATS, fill_value=0)
                .reset_index(name="n")
                .assign(year=self.year, condition=condition)
            )[["condition", "year", "age_cat", "n"]].astype(
                {"year": "int16", "age_cat": "int8", "n": "int32"}
            )
            self.stats.incidence_out_care = pd.concat(  # type: ignore[attr-defined]
                [self.stats.incidence_out_care, incidence_out_care]
            )

    def update_mm(self) -> None:
        """
        Calculate and update the multimorbidity, defined as the number of stage 2 and 3
        comorbidities in each agent.
        """
        self.population["mm"] = self.population[STAGE2 + STAGE3].sum(axis=1)

    def record_stats(self) -> None:
        """ "Record in care age breakdown, out of care age breakdown, reengaging pop age breakdown,
        leaving care age breakdown, and CD4 statistics for both in and out of care populations. 
        If it is an Aim 2 simulation, record the prevalence of all comorbidities, and the 
        multimorbidity for the in care, out of care, initiating, and dying populations. 
        Record the detailed comorbidity information if the multimorbidity detail flag is set.
        """
        stay_in_care = self.population["status"] == ART_USER
        stay_out_care = self.population["status"] == ART_NONUSER
        reengaged = self.population["status"] == REENGAGED
        ltfu = self.population["status"] == LTFU
        dying_art_user = self.population["status"] == DYING_ART_USER
        dying_art_nonuser = self.population["status"] == DYING_ART_NONUSER
        dying = dying_art_nonuser | dying_art_user
        in_care = stay_in_care | ltfu | dying_art_user
        out_care = stay_out_care | reengaged | dying_art_nonuser
        initiating = self.population["h1yy"] == self.year

        # Count of those in care by age and year
        in_care_age = (
            self.population.loc[in_care]
            .groupby(["age"])
            .size()
            .reindex(index=self.parameters.AGES, fill_value=0)
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age", "n"]].astype({"year": "int16", "age": "int8", "n": "int32"})
        self.stats.in_care_age = pd.concat([self.stats.in_care_age, in_care_age])  # type: ignore[attr-defined]

        # Count of those in care by age and year
        out_care_age = (
            self.population.loc[out_care]
            .groupby(["age"])
            .size()
            .reindex(index=self.parameters.AGES, fill_value=0)
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age", "n"]].astype({"year": "int16", "age": "int8", "n": "int32"})
        self.stats.out_care_age = pd.concat([self.stats.out_care_age, out_care_age])  # type: ignore[attr-defined]

        # Count of those reengaging in care by age and year
        reengaged_age = (
            self.population.loc[reengaged]
            .groupby(["age"])
            .size()
            .reindex(index=self.parameters.AGES, fill_value=0)
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age", "n"]].astype({"year": "int16", "age": "int8", "n": "int32"})
        self.stats.reengaged_age = pd.concat([self.stats.reengaged_age, reengaged_age])  # type: ignore[attr-defined]

        # Count of those lost to care by age and year
        ltfu_age = (
            self.population.loc[ltfu]
            .groupby(["age"])
            .size()
            .reindex(index=self.parameters.AGES, fill_value=0)
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age", "n"]].astype({"year": "int16", "age": "int8", "n": "int32"})
        self.stats.ltfu_age = pd.concat([self.stats.ltfu_age, ltfu_age])  # type: ignore[attr-defined]

        # Discretize cd4 count and count those in care
        cd4_in_care = pd.DataFrame(
            np.power(self.population.loc[in_care, "time_varying_sqrtcd4n"], 2).round(0).astype(int)
        ).rename(columns={"time_varying_sqrtcd4n": "cd4_count"})
        cd4_in_care = cd4_in_care.groupby("cd4_count").size()
        cd4_in_care = (
            cd4_in_care.reset_index(name="n")
            .assign(year=self.year)[["year", "cd4_count", "n"]]
            .astype({"year": "int16", "cd4_count": "int16", "n": "int32"})
        )
        self.stats.cd4_in_care = pd.concat([self.stats.cd4_in_care, cd4_in_care])  # type: ignore[attr-defined]

        # Discretize cd4 count and count those out care
        cd4_out_care = pd.DataFrame(
            np.power(self.population.loc[out_care, "time_varying_sqrtcd4n"], 2)
            .round(0)
            .astype(int)
        ).rename(columns={"time_varying_sqrtcd4n": "cd4_count"})
        cd4_out_care = cd4_out_care.groupby("cd4_count").size()
        cd4_out_care = (
            cd4_out_care.reset_index(name="n")
            .assign(year=self.year)[["year", "cd4_count", "n"]]
            .astype({"year": "int16", "cd4_count": "int16", "n": "int32"})
        )
        self.stats.cd4_out_care = pd.concat([self.stats.cd4_out_care, cd4_out_care])  # type: ignore[attr-defined]

        for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
            has_condition = self.population[condition] == 1

            # Record prevalence for in care, out of care, initiator, and dead populations
            prevalence_in_care = (
                self.population.loc[in_care & has_condition]
                .groupby(["age_cat"])
                .size()
                .reindex(index=self.parameters.AGE_CATS, fill_value=0)
                .reset_index(name="n")
                .assign(year=self.year, condition=condition)
            )[["condition", "year", "age_cat", "n"]].astype(
                {"condition": str, "year": "int16", "age_cat": "int8", "n": "int32"}
            )
            self.stats.prevalence_in_care = pd.concat(  # type: ignore[attr-defined]
                [self.stats.prevalence_in_care, prevalence_in_care]
            )
            prevalence_out_care = (
                self.population.loc[out_care & has_condition]
                .groupby(["age_cat"])
                .size()
                .reindex(index=self.parameters.AGE_CATS, fill_value=0)
                .reset_index(name="n")
                .assign(year=self.year, condition=condition)
            )[["condition", "year", "age_cat", "n"]].astype(
                {"condition": str, "year": "int16", "age_cat": "int8", "n": "int32"}
            )
            self.stats.prevalence_out_care = pd.concat(  # type: ignore[attr-defined]
                [self.stats.prevalence_out_care, prevalence_out_care]
            )
            prevalence_inits = (
                self.population.loc[initiating & has_condition]
                .groupby(["age_cat"])
                .size()
                .reindex(index=self.parameters.AGE_CATS, fill_value=0)
                .reset_index(name="n")
                .assign(year=self.year, condition=condition)
            )[["condition", "year", "age_cat", "n"]].astype(
                {"condition": str, "year": "int16", "age_cat": "int8", "n": "int32"}
            )
            self.stats.prevalence_inits = pd.concat(  # type: ignore[attr-defined]
                [self.stats.prevalence_inits, prevalence_inits]
            )
            prevalence_dead = (
                self.population.loc[dying & has_condition]
                .groupby(["age_cat"])
                .size()
                .reindex(index=self.parameters.AGE_CATS, fill_value=0)
                .reset_index(name="n")
                .assign(year=self.year, condition=condition)
            )[["condition", "year", "age_cat", "n"]].astype(
                {"condition": str, "year": "int16", "age_cat": "int8", "n": "int32"}
            )
            self.stats.prevalence_dead = pd.concat(  # type: ignore[attr-defined]
                [self.stats.prevalence_dead, prevalence_dead]
            )

        # Record the multimorbidity information for the in care, out of care, initiating, and dead 
        # populations
        mm_in_care = (
            self.population.loc[in_care]
            .groupby(["age_cat", "mm"])
            .size()
            .reindex(
                index=pd.MultiIndex.from_product(
                    [self.parameters.AGE_CATS, np.arange(0, 8)],
                    names=["age_cat", "mm"],
                ),
                fill_value=0,
            )
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age_cat", "mm", "n"]].astype(
            {"year": "int16", "age_cat": "int8", "mm": "int16", "n": "int32"}
        )
        self.stats.mm_in_care = pd.concat([self.stats.mm_in_care, mm_in_care])  # type: ignore[attr-defined]
        mm_out_care = (
            self.population.loc[out_care]
            .groupby(["age_cat", "mm"])
            .size()
            .reindex(
                index=pd.MultiIndex.from_product(
                    [self.parameters.AGE_CATS, np.arange(0, 8)],
                    names=["age_cat", "mm"],
                ),
                fill_value=0,
            )
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age_cat", "mm", "n"]].astype(
            {"year": "int16", "age_cat": "int8", "mm": "int16", "n": "int32"}
        )
        self.stats.mm_out_care = pd.concat([self.stats.mm_out_care, mm_out_care])  # type: ignore[attr-defined]
        mm_inits = (
            self.population.loc[initiating]
            .groupby(["age_cat", "mm"])
            .size()
            .reindex(
                index=pd.MultiIndex.from_product(
                    [self.parameters.AGE_CATS, np.arange(0, 8)],
                    names=["age_cat", "mm"],
                ),
                fill_value=0,
            )
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age_cat", "mm", "n"]].astype(
            {"year": "int16", "age_cat": "int8", "mm": "int16", "n": "int32"}
        )
        self.stats.mm_inits = pd.concat([self.stats.mm_inits, mm_inits])  # type: ignore[attr-defined]
        mm_dead = (
            self.population.loc[dying]
            .groupby(["age_cat", "mm"])
            .size()
            .reindex(
                index=pd.MultiIndex.from_product(
                    [self.parameters.AGE_CATS, np.arange(0, 8)],
                    names=["age_cat", "mm"],
                ),
                fill_value=0,
            )
            .reset_index(name="n")
            .assign(year=self.year)
        )[["year", "age_cat", "mm", "n"]].astype(
            {"year": "int16", "age_cat": "int8", "mm": "int16", "n": "int32"}
        )
        self.stats.mm_dead = pd.concat([self.stats.mm_dead, mm_dead])  # type: ignore[attr-defined]

        # Record the detailed comorbidity information
        mm_detail_in_care = create_mm_detail_stats(self.population.loc[in_care].copy())
        mm_detail_in_care = mm_detail_in_care.assign(year=self.year)[
            ["year", *ALL_COMORBIDITIES, "n"]
        ].astype({"year": "int16", "n": "int32"})
        self.stats.mm_detail_in_care = pd.concat(  # type: ignore[attr-defined]
            [self.stats.mm_detail_in_care, mm_detail_in_care]
        )

        mm_detail_out_care = create_mm_detail_stats(self.population.loc[out_care].copy())
        mm_detail_out_care = mm_detail_out_care.assign(year=self.year)[
            ["year", *ALL_COMORBIDITIES, "n"]
        ].astype({"year": "int16", "n": "int32"})
        self.stats.mm_detail_out_care = pd.concat(  # type: ignore[attr-defined]
            [self.stats.mm_detail_out_care, mm_detail_out_care]
        )

        mm_detail_inits = create_mm_detail_stats(self.population.loc[initiating].copy())
        mm_detail_inits = mm_detail_inits.assign(year=self.year)[
            ["year", *ALL_COMORBIDITIES, "n"]
        ].astype({"year": "int16", "n": "int32"})
        self.stats.mm_detail_inits = pd.concat([self.stats.mm_detail_inits, mm_detail_inits])  # type: ignore[attr-defined]

        mm_detail_dead = create_mm_detail_stats(self.population.loc[dying].copy())
        mm_detail_dead = mm_detail_dead.assign(year=self.year)[
            ["year", *ALL_COMORBIDITIES, "n"]
        ].astype({"year": "int16", "n": "int32"})
        self.stats.mm_detail_dead = pd.concat([self.stats.mm_detail_dead, mm_detail_dead])  # type: ignore[attr-defined]

    def record_final_stats(self) -> None:
        """all of these are summarized as frequency of events at different tiers, where the last 
        column in the dataset is n. Record some stats that are better calculated at the end of the 
        simulation. A count of new initiators, those dying in care, and those dying out of care is 
        recorded as well as the cd4 count of ART initiators.
        """
        if self.parameters.bmi_intervention:
            """
            bmi_int_cascade: summary statistics on population receiving the intervention and their 
            characteristics
            """
            # record agegroup at art_initiation
            bins = [0, 25, 35, 45, 55, 65, 75, float("inf")]
            # labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
            self.population["init_age_group"] = pd.cut(
                self.population["init_age"], labels=False, bins=bins, right=False
            ).astype("int8")
            # choose columns, fill Na values with 0 and transform to integer
            bmi_int_cascade = self.population[
                [
                    "bmiInt_scenario",
                    "h1yy",
                    "bmiInt_ineligible_dm",
                    "bmiInt_ineligible_underweight",
                    "bmiInt_ineligible_obese",
                    "bmiInt_eligible",
                    "bmiInt_received",
                    "bmi_increase_postART",
                    "bmi_increase_postART_over5p",
                    "become_obese_postART",
                    "bmiInt_impacted",
                ]
            ]

            # Group by all categories and calculate the count in each one
            bmi_int_cascade_count = (
                bmi_int_cascade.groupby(
                    [
                        "bmiInt_scenario",
                        "h1yy",
                        "bmiInt_ineligible_dm",
                        "bmiInt_ineligible_underweight",
                        "bmiInt_ineligible_obese",
                        "bmiInt_eligible",
                        "bmiInt_received",
                        "bmi_increase_postART",
                        "bmi_increase_postART_over5p",
                        "become_obese_postART",
                        "bmiInt_impacted",
                    ]
                )
                .size()
                .reset_index(name="n")
                .astype(
                    {
                        "bmiInt_scenario": "int8",
                        "h1yy": "int16",
                        "bmiInt_ineligible_dm": "bool",
                        "bmiInt_ineligible_underweight": "bool",
                        "bmiInt_ineligible_obese": "bool",
                        "bmiInt_eligible": "bool",
                        "bmiInt_received": "bool",
                        "bmi_increase_postART": "bool",
                        "bmi_increase_postART_over5p": "bool",
                        "become_obese_postART": "bool",
                        "bmiInt_impacted": "bool",
                        "n": "int32",
                    }
                )
            )
            self.stats.bmi_int_cascade = bmi_int_cascade_count  # type: ignore[attr-defined]

            """
            bmi_int_dm_prev: report the number of people with diabetes based on intervention status
            """

            dm_final_output = (
                self.population.groupby(
                    [
                        "bmiInt_scenario",
                        "h1yy",
                        "bmiInt_eligible",
                        "bmiInt_received",
                        "bmiInt_impacted",
                        "dm",
                        "t_dm",
                    ]
                )
                .size()
                .reset_index(name="n")
                .astype(
                    {
                        "bmiInt_scenario": "int8",
                        "h1yy": "int16",
                        "bmiInt_eligible": "bool",
                        "bmiInt_received": "bool",
                        "bmiInt_impacted": "bool",
                        "dm": "bool",
                        "t_dm": "int16",
                        "n": "int32",
                    }
                )
            )

            self.stats.dm_final_output = dm_final_output  # type: ignore[attr-defined]

        dead_in_care = self.population["status"] == DEAD_ART_USER
        dead_out_care = self.population["status"] == DEAD_ART_NONUSER
        new_inits = self.population["h1yy"] >= 2010

        # Count of new initiators by year and age
        new_init_age = self.population.loc[new_inits].groupby(["h1yy", "init_age"]).size()
        new_init_age = new_init_age.reindex(
            pd.MultiIndex.from_product(
                [self.parameters.SIMULATION_YEARS, self.parameters.AGES],
                names=["year", "age"],
            ),
            fill_value=0,
        )
        self.stats.new_init_age = new_init_age.reset_index(name="n").astype(  # type: ignore[attr-defined]
            {"year": "int16", "age": "int8", "n": "int32"}
        )

        # Count of those that died in care by age and year
        dead_in_care_age = self.population.loc[dead_in_care].groupby(["year_died", "age"]).size()
        dead_in_care_age = dead_in_care_age.reindex(
            pd.MultiIndex.from_product(
                [self.parameters.SIMULATION_YEARS, self.parameters.AGES],
                names=["year", "age"],
            ),
            fill_value=0,
        )
        self.stats.dead_in_care_age = dead_in_care_age.reset_index(name="n").astype(  # type: ignore[attr-defined]
            {"year": "int16", "age": "int8", "n": "int32"}
        )

        # Count of those that died out of care by age and year
        dead_out_care_age = self.population.loc[dead_out_care].groupby(["year_died", "age"]).size()
        dead_out_care_age = dead_out_care_age.reindex(
            pd.MultiIndex.from_product(
                [self.parameters.SIMULATION_YEARS, self.parameters.AGES],
                names=["year", "age"],
            ),
            fill_value=0,
        )
        self.stats.dead_out_care_age = dead_out_care_age.reset_index(name="n").astype(  # type: ignore[attr-defined]
            {"year": "int16", "age": "int8", "n": "int32"}
        )

        # Count of discretized cd4 count at ART initiation
        cd4_inits = self.population[["init_sqrtcd4n", "h1yy"]].copy()
        cd4_inits["cd4_count"] = np.power(cd4_inits["init_sqrtcd4n"], 2).round(0).astype(int)
        cd4_inits = cd4_inits.groupby(["h1yy", "cd4_count"]).size()
        self.stats.cd4_inits = (  # type: ignore[attr-defined]
            cd4_inits.reset_index(name="n")
            .rename(columns={"h1yy": "year"})
            .astype({"cd4_count": "int16", "n": "int32"})
        )

        pre_art_bmi = self.population[["pre_art_bmi", "h1yy"]].round(0).astype(int)
        pre_art_bmi = pre_art_bmi.groupby(["h1yy", "pre_art_bmi"]).size()
        self.stats.pre_art_bmi = (  # type: ignore[attr-defined]
            pre_art_bmi.reset_index(name="n")
            .rename(columns={"h1yy": "year"})
            .astype({"year": "int16", "pre_art_bmi": "int8", "n": "int32"})
        )

        post_art_bmi = self.population[["post_art_bmi", "h1yy", "pre_art_bmi", "bmiInt_scenario"]]

        # post_art_bmi are break into categories instead of report the exactly number of BMI
        post_art_bmi.assign(pre_bmi_cat=np.array(1, dtype="int8"))
        post_art_bmi.loc[post_art_bmi["pre_art_bmi"] < 18.5, "pre_bmi_cat"] = np.array(
            0, dtype="int8"
        )
        post_art_bmi.loc[post_art_bmi["pre_art_bmi"] >= 30, "pre_bmi_cat"] = np.array(
            2, dtype="int8"
        )

        post_art_bmi.assign(post_bmi_cat=np.array(1, dtype="int8"))
        post_art_bmi.loc[post_art_bmi["post_art_bmi"] < 18.5, "post_bmi_cat"] = np.array(
            0, dtype="int8"
        )
        post_art_bmi.loc[post_art_bmi["post_art_bmi"] >= 30, "post_bmi_cat"] = np.array(
            2, dtype="int8"
        )

        post_art_bmi = post_art_bmi.groupby(
            ["bmiInt_scenario", "h1yy", "post_bmi_cat", "pre_bmi_cat"]
        ).size()

        self.stats.post_art_bmi = (  # type: ignore[attr-defined]
            post_art_bmi.reset_index(name="n")
            .rename(columns={"h1yy": "year"})
            .astype({"n": "int32"})
        )


###############################################################################
#     Statistics Classes                                                      #
###############################################################################


class Statistics:
    """A class housing the output from a PEARL run."""

    def __init__(
        self,
        output_folder: Path,
        group_name: str,
        replication: int,
    ) -> None:
        """The init function operates on two levels. If called with no out_list a new Statistics 
        class is initialized, with empty dataframes to fill with data. Otherwise it concatenates 
        the out_list dataframes so that the results of all replications and groups are stored in 
        a single dataframe.
        """

        self.output_folder = output_folder
        self.group_name = group_name
        self.replication = replication

    def __getattr__(self, attr: str) -> None:
        """
        If an attribute is not present and it is called for, default to an empty pandas dataframe
        """
        setattr(self, attr, pd.DataFrame())

    def save(self) -> None:
        """Save all internal dataframes as parquet files."""
        for name, item in self.__dict__.items():
            if isinstance(item, pd.DataFrame):
                try:
                    item = item.assign(group=self.group_name, replication=self.replication).astype(
                        {"replication": "int16"}
                    )
                    item.to_parquet(self.output_folder / f"{name}.parquet", index=False)
                except Exception as e:
                    print(f"Error saving DataFrame {name}: {e}")
