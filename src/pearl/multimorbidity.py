"""
Functions pertraining to multimorbidity calculations
"""

from typing import Any

import numpy as np
import pandas as pd

from pearl.definitions import ART_NONUSER, STAGE0, STAGE1, STAGE2, STAGE3
from pearl.interpolate import restricted_cubic_spline_var
from pearl.parameters import Parameters


def create_mm_detail_stats(pop: pd.DataFrame) -> pd.DataFrame:
    all_comorbidities = STAGE0 + STAGE1 + STAGE2 + STAGE3
    df = pop[["age_cat"] + all_comorbidities].copy()
    # Count how many people have each unique set of comorbidities
    df = df.groupby(all_comorbidities).size().reset_index(name="n")
    return df


def create_comorbidity_pop_matrix(
    pop: pd.DataFrame, condition: str, parameters: Parameters
) -> Any:
    """
    Create and return the population matrix as a numpy array for calculating the probability of
    incidence of any of the 9 comorbidities. Each comorbidity has a unique set of variables as
    listed below.

    Parameters
    ----------
    pop : pd.DataFrame
        Population for use in preparation for Pearl.calculate_prob.
    condition : str
        Morbidity as a string value.
    parameters : Parameters
        Parameters object that contains spline knots for interpolation calculations for select
        morbidities.

    Returns
    -------
    NDArray[Any]
        numpy array representing the population for use in Pearl.calculate_prob.
    """
    # Calculate some intermediate variables needed for the population matrix
    pop["time_since_art"] = pop["year"] - pop["h1yy"]
    pop["out_care"] = (pop["status"] == ART_NONUSER).astype(int)
    if condition in STAGE2 + STAGE3:
        pop["delta_bmi_"] = restricted_cubic_spline_var(
            pop["delta_bmi"].to_numpy(), parameters.delta_bmi_dict[condition].to_numpy(), 1
        )
        pop["delta_bmi__"] = restricted_cubic_spline_var(
            pop["delta_bmi"].to_numpy(), parameters.delta_bmi_dict[condition].to_numpy(), 2
        )
        pop["post_art_bmi_"] = restricted_cubic_spline_var(
            pop["post_art_bmi"].to_numpy(), parameters.post_art_bmi_dict[condition].to_numpy(), 1
        )
        pop["post_art_bmi__"] = restricted_cubic_spline_var(
            pop["post_art_bmi"].to_numpy(), parameters.post_art_bmi_dict[condition].to_numpy(), 2
        )

    if condition == "anx":
        return pop[
            [
                "age",
                "init_sqrtcd4n",
                "dpr",
                "time_since_art",
                "hcv",
                "intercept",
                "smoking",
                "year",
            ]
        ].to_numpy(dtype=float)
    elif condition == "dpr":
        return pop[
            [
                "age",
                "anx",
                "init_sqrtcd4n",
                "time_since_art",
                "hcv",
                "intercept",
                "smoking",
                "year",
            ]
        ].to_numpy(dtype=float)
    elif condition == "ckd":
        return pop[
            [
                "age",
                "anx",
                "delta_bmi_",
                "delta_bmi__",
                "delta_bmi",
                "post_art_bmi",
                "post_art_bmi_",
                "post_art_bmi__",
                "init_sqrtcd4n",
                "dm",
                "dpr",
                "time_since_art",
                "hcv",
                "ht",
                "intercept",
                "lipid",
                "smoking",
                "year",
            ]
        ].to_numpy(dtype=float)
    elif condition == "lipid":
        return pop[
            [
                "age",
                "anx",
                "delta_bmi_",
                "delta_bmi__",
                "delta_bmi",
                "post_art_bmi",
                "post_art_bmi_",
                "post_art_bmi__",
                "init_sqrtcd4n",
                "ckd",
                "dm",
                "dpr",
                "time_since_art",
                "hcv",
                "ht",
                "intercept",
                "smoking",
                "year",
            ]
        ].to_numpy(dtype=float)
    elif condition == "dm":
        return pop[
            [
                "age",
                "anx",
                "delta_bmi_",
                "delta_bmi__",
                "delta_bmi",
                "post_art_bmi",
                "post_art_bmi_",
                "post_art_bmi__",
                "init_sqrtcd4n",
                "ckd",
                "dpr",
                "time_since_art",
                "hcv",
                "ht",
                "intercept",
                "lipid",
                "smoking",
                "year",
            ]
        ].to_numpy(dtype=float)
    elif condition == "ht":
        return pop[
            [
                "age",
                "anx",
                "delta_bmi_",
                "delta_bmi__",
                "delta_bmi",
                "post_art_bmi",
                "post_art_bmi_",
                "post_art_bmi__",
                "init_sqrtcd4n",
                "ckd",
                "dm",
                "dpr",
                "time_since_art",
                "hcv",
                "intercept",
                "lipid",
                "smoking",
                "year",
            ]
        ].to_numpy(dtype=float)
    elif condition in ["malig", "esld", "mi"]:
        return pop[
            [
                "age",
                "anx",
                "delta_bmi_",
                "delta_bmi__",
                "delta_bmi",
                "post_art_bmi",
                "post_art_bmi_",
                "post_art_bmi__",
                "init_sqrtcd4n",
                "ckd",
                "dm",
                "dpr",
                "time_since_art",
                "hcv",
                "ht",
                "intercept",
                "lipid",
                "smoking",
                "year",
            ]
        ].to_numpy(dtype=float)
    else:
        return np.array([], dtype=float)
