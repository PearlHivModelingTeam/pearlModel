"""
Parameters class
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pearl.definitions import STAGE0, STAGE1, STAGE2, STAGE3


class Parameters:
    """This class holds all the parameters needed for PEARL to run."""

    def __init__(
        self,
        path: Path,
        rerun_folder: Path,
        output_folder: Path,
        group_name: str,
        new_dx: str,
        final_year: int,
        mortality_model: str,
        mortality_threshold_flag: bool,
        idu_threshold: str,
        verbose: bool,
        seed: int,
        bmi_intervention: int = 0,
        bmi_intervention_scenario: int = 1,
        bmi_intervention_start_year: int = 2020,
        bmi_intervention_end_year: int = 2030,
        bmi_intervention_coverage: float = 1.0,
        bmi_intervention_effectiveness: float = 1.0,
    ):
        """Takes the path to the parameters.h5 file, the path to the folder containing rerun data 
        if the run is a rerun, the output folder, the group name, a flag indicating if the 
        simulation is for aim 2, a flag indicating whether to record detailed comorbidity 
        information, the type of new_dx parameter to use, the final year of the model, the 
        mortality model to use, whether to use a mortality threshold, verbosity, the sensitivity 
        analysis dict, the classic sensitivity analysis dict, and the aim 2 sensitivity 
        analysis dict.
        """

        # Save inputs as class attributes
        self.rerun_folder = rerun_folder
        self.output_folder = output_folder
        self.group_name = group_name
        self.final_year = final_year
        self.mortality_threshold_flag = mortality_threshold_flag
        self.verbose = verbose
        self.seed = seed

        # 2009 population
        self.on_art_2009 = pd.read_hdf(path, "on_art_2009").loc[group_name]
        self.age_in_2009 = pd.read_hdf(path, "age_in_2009").loc[group_name]
        self.h1yy_by_age_2009 = pd.read_hdf(path, "h1yy_by_age_2009").loc[group_name]
        self.cd4n_by_h1yy_2009 = pd.read_hdf(path, "cd4n_by_h1yy_2009").loc[group_name]

        # New initiator statistics
        self.linkage_to_care = pd.read_hdf(path, "linkage_to_care").loc[group_name]
        self.age_by_h1yy = pd.read_hdf(path, "age_by_h1yy").loc[group_name]
        self.cd4n_by_h1yy = pd.read_hdf(path, "cd4n_by_h1yy").loc[group_name]

        # Choose new ART initiator model
        if new_dx == "base":
            self.new_dx = pd.read_hdf(path, "new_dx").loc[group_name]
        elif new_dx == "ehe":
            self.new_dx = pd.read_hdf(path, "new_dx_ehe").loc[group_name]
        else:
            raise ValueError("Invalid new diagnosis file specified")

        # Choose mortality model
        if mortality_model == "by_sex_race_risk":
            mortality_model_str = ""
        else:
            mortality_model_str = "_" + mortality_model

        if (mortality_model != "by_sex_race_risk") and (
            mortality_model != "by_sex_race_risk_2015" and (idu_threshold != "2x")
        ):
            raise NotImplementedError(
                "Alternative mortality models with idu threshold changes is not implemented"
            )

        # Mortality In Care
        self.mortality_in_care = pd.read_hdf(path, f"mortality_in_care{mortality_model_str}").loc[
            group_name
        ]
        self.mortality_in_care_age = pd.read_hdf(
            path, f"mortality_in_care_age{mortality_model_str}"
        ).loc[group_name]
        self.mortality_in_care_sqrtcd4 = pd.read_hdf(
            path, f"mortality_in_care_sqrtcd4{mortality_model_str}"
        ).loc[group_name]
        self.mortality_in_care_vcov = pd.read_hdf(path, "mortality_in_care_vcov").loc[group_name]

        # Mortality Out Of Care
        self.mortality_out_care = pd.read_hdf(
            path, f"mortality_out_care{mortality_model_str}"
        ).loc[group_name]
        self.mortality_out_care_age = pd.read_hdf(
            path, f"mortality_out_care_age{mortality_model_str}"
        ).loc[group_name]
        self.mortality_out_care_tv_sqrtcd4 = pd.read_hdf(
            path, f"mortality_out_care_tv_sqrtcd4{mortality_model_str}"
        ).loc[group_name]
        self.mortality_out_care_vcov = pd.read_hdf(path, "mortality_out_care_vcov").loc[group_name]

        # Mortality Threshold
        if idu_threshold != "2x":
            self.mortality_threshold = pd.read_hdf(
                path, f"mortality_threshold_idu_{idu_threshold}"
            ).loc[group_name]
        else:
            self.mortality_threshold = pd.read_hdf(
                path, f"mortality_threshold{mortality_model_str}"
            ).loc[group_name]

        # Loss To Follow Up
        self.loss_to_follow_up = pd.read_hdf(path, "loss_to_follow_up").loc[group_name]
        self.ltfu_knots = pd.read_hdf(path, "ltfu_knots").loc[group_name]
        self.loss_to_follow_up_vcov = pd.read_hdf(path, "loss_to_follow_up_vcov").loc[group_name]

        # Cd4 Increase
        self.cd4_increase = pd.read_hdf(path, "cd4_increase").loc[group_name]
        self.cd4_increase_knots = pd.read_hdf(path, "cd4_increase_knots").loc[group_name]
        self.cd4_increase_vcov = pd.read_hdf(path, "cd4_increase_vcov").loc[group_name]

        # Cd4 Decrease
        self.cd4_decrease = pd.read_hdf(path, "cd4_decrease").loc["all"]
        self.cd4_decrease_vcov = pd.read_hdf(path, "cd4_decrease_vcov")

        # Years out of Care
        self.years_out_of_care = pd.read_hdf(path, "years_out_of_care")

        # BMI
        self.pre_art_bmi = pd.read_hdf(path, "pre_art_bmi").loc[group_name]
        self.pre_art_bmi_model = pd.read_hdf(path, "pre_art_bmi_model").loc[group_name].values[0]
        self.pre_art_bmi_age_knots = pd.read_hdf(path, "pre_art_bmi_age_knots").loc[group_name]
        self.pre_art_bmi_h1yy_knots = pd.read_hdf(path, "pre_art_bmi_h1yy_knots").loc[group_name]
        self.pre_art_bmi_rse = pd.read_hdf(path, "pre_art_bmi_rse").loc[group_name].values[0]
        self.post_art_bmi = pd.read_hdf(path, "post_art_bmi").loc[group_name]
        self.post_art_bmi_age_knots = pd.read_hdf(path, "post_art_bmi_age_knots").loc[group_name]
        self.post_art_bmi_pre_art_bmi_knots = pd.read_hdf(
            path, "post_art_bmi_pre_art_bmi_knots"
        ).loc[group_name]
        self.post_art_bmi_cd4_knots = pd.read_hdf(path, "post_art_bmi_cd4_knots").loc[group_name]
        self.post_art_bmi_cd4_post_knots = pd.read_hdf(path, "post_art_bmi_cd4_post_knots").loc[
            group_name
        ]
        self.post_art_bmi_rse = pd.read_hdf(path, "post_art_bmi_rse").loc[group_name].values[0]

        # BMI Intervention Probability
        self.bmi_intervention = bmi_intervention
        self.bmi_intervention_scenario = bmi_intervention_scenario
        self.bmi_intervention_start_year = bmi_intervention_start_year
        self.bmi_intervention_end_year = bmi_intervention_end_year
        self.bmi_intervention_coverage = bmi_intervention_coverage
        self.bmi_intervention_effectiveness = bmi_intervention_effectiveness

        # Comorbidities
        self.prev_users_dict = {
            comorbidity: pd.read_hdf(path, f"{comorbidity}_prev_users").loc[group_name]
            for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
        }
        self.prev_inits_dict = {
            comorbidity: pd.read_hdf(path, f"{comorbidity}_prev_inits").loc[group_name]
            for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
        }
        self.comorbidity_coeff_dict = {
            comorbidity: pd.read_hdf(path, f"{comorbidity}_coeff").loc[group_name]
            for comorbidity in STAGE1 + STAGE2 + STAGE3
        }
        self.delta_bmi_dict = {
            comorbidity: pd.read_hdf(path, f"{comorbidity}_delta_bmi").loc[group_name]
            for comorbidity in STAGE2 + STAGE3
        }
        self.post_art_bmi_dict = {
            comorbidity: pd.read_hdf(path, f"{comorbidity}_post_art_bmi").loc[group_name]
            for comorbidity in STAGE2 + STAGE3
        }

        # Aim 2 Mortality
        self.mortality_in_care_co = pd.read_hdf(path, "mortality_in_care_co").loc[group_name]
        self.mortality_in_care_post_art_bmi = pd.read_hdf(
            path, "mortality_in_care_post_art_bmi"
        ).loc[group_name]
        self.mortality_out_care_co = pd.read_hdf(path, "mortality_out_care_co").loc[group_name]
        self.mortality_out_care_post_art_bmi = pd.read_hdf(
            path, "mortality_out_care_post_art_bmi"
        ).loc[group_name]

        # Year and age ranges
        self.AGES = np.arange(18, 87)
        self.AGE_CATS = np.arange(2, 8)
        self.SIMULATION_YEARS = np.arange(2010, final_year + 1)
        self.ALL_YEARS = np.arange(2000, final_year + 1)
        self.INITIAL_YEARS = np.arange(2000, 2010)
        self.CD4_BINS = np.arange(2001)
