"""
Module containing the Pearl class for simulation and the Statitistics class for storing output
values.
"""

import os
from typing import Dict

# TODO move this somewhere better, like into docker
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pearl.definitions import (
    ALL_COMORBIDITIES,
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
)
from pearl.interpolate import restricted_quadratic_spline_var
from pearl.multimorbidity import create_comorbidity_pop_matrix, create_mm_detail_stats
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
)
from pearl.sample import draw_from_trunc_norm

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.options.mode.chained_assignment = None  # default='warn'

###############################################################################
# Pearl Class                                                                 #
###############################################################################


class Pearl:
    """
    Class containing the pearl simulation engine.
    """

    def __init__(self, parameters: Parameters):
        """
        Takes an instance of the Parameters class and
        runs a simulation.

        Parameters
        ----------
        parameters : Parameters
            Parameters object which loads and stores all model values.
        """
        self.parameters = parameters
        self.group_name = self.parameters.group_name
        self.replication = self.parameters.replication
        self.year = 2009
        self.random_state = self.parameters.random_state

        if self.parameters.output_folder:
            with Path.open(self.parameters.output_folder / "random.state", "w") as state_file:
                state_file.write(str(self.parameters.seed))

        # Initiate output class
        self.stats = Statistics(
            output_folder=self.parameters.output_folder,
            group_name=self.group_name,
            replication=self.replication,
        )

        # Create art using 2009 population
        n_initial_users = self.parameters.on_art_2009.iloc[0]
        user_pop = self.make_user_pop_2009(n_initial_users)

        # Create art non-using 2009 population
        non_user_pop = self.make_nonuser_pop_2009(self.parameters.n_initial_nonusers)

        # concat to get initial population
        self.population = pd.concat([user_pop, non_user_pop])

        # Create population of new art initiators
        art_pop = self.make_new_population(self.parameters.n_new_agents, self.random_state)

        # concat the art pop to population
        self.population = pd.concat([self.population, art_pop]).fillna(0)

        self.population = self.population.reset_index()
        self.population["id"] = np.array(range(self.population.index.size))
        self.population = self.population.set_index(["id"])

        if self.parameters.history:
            self.population.to_parquet(
                self.parameters.output_folder / "history.parquet", compression="zstd"
            )
            # First recording of stats
            self.record_stats()

        # Move to 2010
        self.year += 1

    @staticmethod
    def calculate_prob(pop: pd.DataFrame, coeffs: NDArray[Any]) -> NDArray[Any]:
        """
        Calculate and return a numpy array of individual probabilities from logistic regression
        given the population and coefficient matrices.
        Used for multiple logistic regression functions.

        Parameters
        ----------
        pop : pd.DataFrame
            Population Dataframe that results from calling create_mortality_in_care_pop_matrix,
            create_mortality_out_care_pop_matrix, create_ltfu_pop_matrix, or
            create_comorbidity_pop_matrix.

        coeffs : NDArray[Any]
            Coefficients are stored in Parameters object with attribute names corresponding to
            the above population preparation functions.

        Returns
        -------
        NDArray[Any]
            Result of multiplying the population by the coefficients and converting to probability.
        """
        # Calculate log odds using a matrix multiplication
        log_odds = np.matmul(pop, coeffs)

        # Convert to probability
        prob = np.exp(log_odds) / (1.0 + np.exp(log_odds))
        return np.array(prob)

    @staticmethod
    def create_ltfu_pop_matrix(pop: pd.DataFrame, knots: pd.DataFrame) -> Any:
        """
        Create and return the population matrix as a numpy array for use in calculating probability
        of loss to follow up.

        Parameters
        ----------
        pop : pd.DataFrame
            The population DataFrame that we wish to calculate loss to follow up on.
        knots : pd.DataFrame
            Quadratic spline knot values which are stored in Parameters.

        Returns
        -------
        NDArray[Any]
            numpy array for passing into Pearl.calculate_prob
        """
        # Create all needed intermediate variables
        knots = knots.to_numpy()
        pop["age_"] = restricted_quadratic_spline_var(pop["age"].to_numpy(), knots, 1)
        pop["age__"] = restricted_quadratic_spline_var(pop["age"].to_numpy(), knots, 2)
        pop["age___"] = restricted_quadratic_spline_var(pop["age"].to_numpy(), knots, 3)
        pop["haart_period"] = (pop["h1yy"].values > 2010).astype(int)
        return pop[
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
        ].to_numpy()

    @staticmethod
    def add_age_categories(population: pd.DataFrame) -> pd.DataFrame:
        """
        Add an age_cat column corresponding to the decade age of the agent, truncated at a maximum
        age category of 7.

        Parameters
        ----------
        population : pd.DataFrame
            Population with an age column.

        Returns
        -------
        pd.DataFrame
            Population with age_cat column added.
        """
        population["age"] = np.floor(population["age"])
        population["age_cat"] = np.floor(population["age"] / 10)
        population.loc[population["age_cat"] > 7, "age_cat"] = 7
        population.loc[population["age_cat"] < 2, "age_cat"] = 2
        population["id"] = np.array(range(population.index.size))
        population = population.set_index(["age_cat", "id"]).sort_index()

        return population

    @staticmethod
    def add_default_columns(population: pd.DataFrame) -> pd.DataFrame:
        """
        Add default values for columns necessary for simulation.

        Parameters
        ----------
        population : pd.DataFrame
            Population DataFrame to add default columns to.

        Returns
        -------
        pd.DataFrame
            Population with added default columns
        """
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

        return population

    def add_h1yy(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        For each age category (age_cat), assign a value for ART initiation (h1yy) based on
        NA-ACCORD data distribution in 2009 cohort.

        Parameters
        ----------
        population : pd.DataFrame
            Population to add h1yy column to.

        Returns
        -------
        pd.DataFrame
            Population with added h1yy column.
        """
        # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
        for age_cat, grouped in population.groupby("age_cat"):
            h1yy_data = self.parameters.h1yy_by_age_2009.loc[age_cat].reset_index()
            population.loc[age_cat, "h1yy"] = self.random_state.choice(
                h1yy_data["h1yy"], size=len(grouped), p=h1yy_data["pct"]
            )

        # Reindex for group operation
        population["h1yy"] = population["h1yy"].astype(int)
        population = population.reset_index().set_index(["h1yy", "id"]).sort_index()

        return population

    def add_init_sqrtcd4n(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        For each unique h1yy value, sample from a truncated normal distribution between 0 and
        sqrt(2000) with mu, and sigma values based on NA-ACCORD data distributions from 2009
        cohort.

        Parameters
        ----------
        population : pd.DataFrame
            Population to add sqrtcd4n column to.

        Returns
        -------
        pd.DataFrame
            Population with sqrtcd4n column added.
        """
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

        return population

    def add_bmi(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Add Pre ART BMI (pre_art_bmi), Post ART BMI (post_art_bmi), and Delta BMI (delta_bmi)
        columns to population Dataframe based on the outputs of
        Pearl.generation.calculate_pre_art_bmi, Pearl.generation.calculate_post_art_bmi, and the
        difference between post_art_bmi and pre_art_bmi.

        Parameters
        ----------
        population : pd.DataFrame
            Population to add BMI data to.

        Returns
        -------
        pd.DataFrame
            Population with added BMI columns.
        """
        population["pre_art_bmi"] = calculate_pre_art_bmi(
            population.copy(), self.parameters, self.random_state
        )
        population["post_art_bmi"] = calculate_post_art_bmi(
            population.copy(), self.parameters, self.random_state
        )
        population["delta_bmi"] = population["post_art_bmi"] - population["pre_art_bmi"]

        return population

    def add_comorbidity(
        self,
        condition: str,
        population: pd.DataFrame,
        condition_probabilities: Dict[str, pd.Series],
        user: bool,
    ) -> pd.DataFrame:
        """
        Add the selected condition based on the probability defined in self.parameters.

        Parameters
        ----------
        condition : str
            Condition to be added.
        population : pd.DataFrame
            Population to add condition to.

        Returns
        -------
        pd.DataFrame
            Population dataframe with condition added.
        """
        morbidity_probability = condition_probabilities[condition].values

        population[condition] = (
            self.random_state.rand(len(population.index)) < morbidity_probability
        ).astype(int)
        if user:
            population[f"t_{condition}"] = np.array(0, dtype="int8")
        else:
            population[f"t_{condition}"] = population[condition]

        return population

    def make_base_population(self, n_population: int) -> pd.DataFrame:
        """
        Create and return initial 2009 population dataframe. Draw ages from a mixed normal
        distribution truncated at 18 and 85. Assign ART initiation year using proportions from
        NA-ACCORD data. Draw sqrt CD4 count from a normal distribution truncated at 0 and
        sqrt(2000).

        Parameters
        ----------
        n_population : int
            Number of agents to generate

        Returns
        -------
        pd.DataFrame
            Population Dataframe for simulation.
        """
        # Create population dataframe
        population = pd.DataFrame()

        # Draw ages from the truncated mixed gaussian
        population["age"] = simulate_ages(
            self.parameters.age_in_2009, n_population, self.random_state
        )

        # Create age categories
        population = self.add_age_categories(population)

        # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
        population = self.add_h1yy(population)

        # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 and sqrt 2000
        population = self.add_init_sqrtcd4n(population)

        # Toss out age_cat < 2
        population.loc[population["age_cat"] < 2, "age_cat"] = 2

        # Add final columns used for calculations and output
        population = self.add_default_columns(population)

        # Calculate time varying cd4 count
        population["time_varying_sqrtcd4n"] = calculate_cd4_increase(
            population.copy(), self.parameters
        )

        return population

    def make_user_pop_2009(self, n_initial_users: int) -> pd.DataFrame:
        """
        Generate base population, assign bmi, comorbidities, and multimorbidity
        using their respective models. Assign status column to ART_USER global value.

        Parameters
        ----------
        n_initial_users : int
            Number of agents to generate that will be counted as ART users in 2009.

        Returns
        -------
        pd.DataFrame
            Population DataFrame representing all ART users in 2009 for the simulation.
        """
        population = self.make_base_population(n_initial_users)

        # Set status and initiate out of care variables
        population["status"] = ART_USER

        # add bmi, comorbidity, and multimorbidity columns

        # Bmi
        population = self.add_bmi(population)

        # Apply comorbidities
        for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
            population = self.add_comorbidity(
                condition, population, self.parameters.prev_users_dict, user=True
            )

        population["mm"] = np.array(population[STAGE2 + STAGE3].sum(axis=1), dtype="int8")

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        population = population.astype(POPULATION_TYPE_DICT)

        return population

    def make_nonuser_pop_2009(self, n_initial_nonusers: int) -> pd.DataFrame:
        """
        Assign bmi, comorbidities, and multimorbidity using their respective models.
        Assign status column to ART_NONUSER global value.

        Parameters
        ----------
        n_initial_nonusers : int
            Number of agents to generate that will be counted as NONART users in 2009.

        Returns
        -------
        pd.DataFrame
            Population DataFrame representing all NONART users in 2009 for the simulation.
        """
        population = self.make_base_population(n_initial_nonusers)

        # Set status and initiate out of care variables
        years_out_of_care = self.random_state.choice(
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
        population = self.add_bmi(population)

        # Apply comorbidities
        for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
            population = self.add_comorbidity(
                condition, population, self.parameters.prev_users_dict, user=False
            )

        population["mm"] = population[STAGE2 + STAGE3].sum(axis=1)

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        population = population.astype(POPULATION_TYPE_DICT)

        return population

    def make_new_population(
        self, n_new_agents: pd.DataFrame, random_state: np.random.RandomState
    ) -> pd.DataFrame:
        """
        Create and return the population initiating ART during the simulation. Age and CD4
        count distribution parameters are taken from a linear regression until 2018 and drawn from
        a uniform distribution between the 2018 values and the predicted values thereafter. Ages
        are drawn from the two-component mixed normal distribution truncated at 18 and 85 defined
        by the generated parameters. The n_new_agents dataframe defines the population size of ART
        initiators and those not initiating immediately. The latter population begins ART some
        years later as drawn from a normalized, truncated Poisson distribution. The sqrt CD4 count
        at ART initiation for each agent is drawn from a normal distribution truncated at 0 and
        sqrt 2000 as defined by the generated parameters. If this is an Aim 2 simulation, generate
        bmi, comorbidities, and multimorbidity from their respective distributions.

        Parameters
        ----------
        n_new_agents : pd.DataFrame
            Number of new agents to generate.
        random_state : np.random.RandomState
            Random State object for random number sampling.

        Returns
        -------
        pd.DataFrame
            Population DataFrame containing all agents initiating ART in the simulation.
        """

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
        population["id"] = np.arange(population.index.size)
        population = population.set_index(["age_cat", "id"]).sort_index()
        population = population.reset_index()
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
        for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
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

        # population[
        #     [
        #         "bmiInt_scenario",
        #         "bmiInt_ineligible_dm",
        #         "bmiInt_ineligible_underweight",
        #         "bmiInt_ineligible_obese",
        #         "bmiInt_eligible",
        #         "bmiInt_received",
        #         "bmi_increase_postART",
        #         "bmi_increase_postART_over5p",
        #         "become_obese_postART",
        #         "bmiInt_impacted",
        #         "pre_art_bmi",
        #         "post_art_bmi_without_bmiInt",
        #         "post_art_bmi",
        #     ]
        # ] = apply_bmi_intervention(population.copy(), self.parameters, random_state)

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
            self.random_state = np.random.RandomState(seed=42)

            # Apply comorbidities
            self.apply_comorbidity_incidence()
            self.random_state = np.random.RandomState(seed=42)

            self.update_mm()
            self.random_state = np.random.RandomState(seed=42)

            # In care operations
            self.increase_cd4_count()  # Increase cd4n in people in care
            self.random_state = np.random.RandomState(seed=42)

            self.add_new_user()  # Add in newly diagnosed ART initiators
            self.random_state = np.random.RandomState(seed=42)

            self.kill_in_care()  # Kill some people in care
            self.random_state = np.random.RandomState(seed=42)

            self.lose_to_follow_up()  # Lose some people to follow up
            self.random_state = np.random.RandomState(seed=42)

            # Out of care operations
            self.decrease_cd4_count()  # Decrease cd4n in people out of care
            self.random_state = np.random.RandomState(seed=42)

            self.kill_out_care()  # Kill some people out of care
            self.random_state = np.random.RandomState(seed=42)

            self.reengage()  # Reengage some people out of care
            self.random_state = np.random.RandomState(seed=42)

            # Append changed populations to their respective DataFrames
            self.append_new()
            self.random_state = np.random.RandomState(seed=42)

            # Increment year
            self.year += 1

            # store history
            if self.parameters.history:
                # Record output statistics
                self.record_stats()
                self.population.to_parquet(
                    self.parameters.output_folder / "history.parquet",
                    engine="fastparquet",
                    append=True,
                    compression="zstd",
                )

        self.population = self.population.assign(
            group=self.group_name, replication=self.replication
        )
        if self.parameters.output_folder and self.parameters.final_state:
            self.population.to_parquet(
                self.parameters.output_folder / "final_state.parquet", compression="zstd"
            )

        # Record output statistics for the end of the simulation
        # self.record_final_stats()

        # Save output
        if self.parameters.output_folder:
            self.stats.save()

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

            if (
                self.parameters.sa_variables
                and f"{condition}_incidence" in self.parameters.sa_variables
            ):
                prob = np.clip(
                    a=prob * self.parameters.sa_scalars[f"{condition}_incidence"],
                    a_min=0,
                    a_max=1,
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

        bmi_int_cascade: summary statistics on population receiving the intervention and their
        characteristics
        """
        # record agegroup at art_initiation
        bins = [0, 25, 35, 45, 55, 65, 75, float("inf")]
        # labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
        self.population["init_age_group"] = pd.cut(
            self.population["init_age"], labels=False, bins=bins, right=False
        ).astype("int8")

        # record bmi group at art_initiation
        pre_art_bmi_bins = [0, 18.5, 21.5, 25, 27.5, 30, float("inf")]
        self.population["init_bmi_group"] = pd.cut(
            self.population["pre_art_bmi"], labels=False, bins=pre_art_bmi_bins, right=False
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
        self.stats.bmi_int_dm_prev = (  # type: ignore[attr-defined]
            self.population.groupby(
                [
                    "bmiInt_scenario",
                    "h1yy",
                    "init_age_group",
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
                    "init_age_group": "int8",
                    "bmiInt_eligible": "bool",
                    "bmiInt_received": "bool",
                    "bmiInt_impacted": "bool",
                    "dm": "bool",
                    "t_dm": "int16",
                    "n": "int32",
                }
            )
        )

        dm_final_output = (
            self.population.groupby(
                [
                    "bmiInt_scenario",
                    "h1yy",
                    "init_age_group",
                    "init_bmi_group",
                    "bmiInt_eligible",
                    "bmiInt_received",
                    "bmiInt_impacted",
                    "dm",
                    "t_dm",
                    "year_died",
                ]
            )
            .size()
            .reset_index(name="n")
            .astype(
                {
                    "bmiInt_scenario": "int8",
                    "h1yy": "int16",
                    "init_age_group": "int8",
                    "init_bmi_group": "int8",
                    "bmiInt_eligible": "bool",
                    "bmiInt_received": "bool",
                    "bmiInt_impacted": "bool",
                    "dm": "bool",
                    "t_dm": "int16",
                    "year_died": "int16",
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
        """
        The init function operates on two levels. If called with no out_list a new Statistics
        class is initialized, with empty dataframes to fill with data. Otherwise it concatenates
        the out_list dataframes so that the results of all replications and groups are stored in
        a single dataframe.

        Parameters
        ----------
        output_folder : Path
            Path to write outputs to.
        group_name : str
            Name of the subpopulation that was simulated, used for creating folder structure in
            output folder.
        replication : int
            Replication number of simulation, user for creating folder struncture in output folder.
        """

        self.output_folder = output_folder
        self.group_name = group_name
        self.replication = replication

    def __getattr__(self, attr: str) -> None:
        """If an attribute is not present and it is called for, default to an empty pandas
        dataframe

        Parameters
        ----------
        attr : str
            Name of attribute.
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
                    item.to_parquet(
                        self.output_folder / f"{name}.parquet", index=False, compression="zstd"
                    )
                except Exception as e:
                    print(f"Error saving DataFrame {name}: {e}")
