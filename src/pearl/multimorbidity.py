'''
Functions pertraining to multimorbidity calculations
'''
import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typing import Any

from pearl.definitions import (STAGE0, STAGE1, STAGE2, STAGE3, ART_NONUSER)
from pearl.interpolate import restricted_cubic_spline_var
from pearl.parameters import Parameters

def create_mm_detail_stats(pop: pd.DataFrame) -> pd.DataFrame:
    """Encode all comorbidity information as an 11 bit integer and return a dataframe counting the 
    number of agents with every unique set of comorbidities.
    """

    all_comorbidities = STAGE0 + STAGE1 + STAGE2 + STAGE3
    df = pop[["age_cat"] + all_comorbidities].copy()
    if not df.empty:
        """This line of code adds a new column 'multimorbidity' to the DataFrame. The column is
        created by applying a function to each row of the DataFrame. This function:
        Converts the values in the comorbidity columns to strings.
        Joins these strings to create a concatenated binary representation for each row.
        Converts the binary representation to an integer using base 2."""
        df["multimorbidity"] = (
            df[all_comorbidities]
            .apply(lambda row: "".join(row.values.astype(int).astype(str)), axis="columns")
            .apply(int, base=2)
            .astype("int32")
        )
    else:
        df["multimorbidity"] = []

    # Count how many people have each unique set of comorbidities
    df = df.groupby(["multimorbidity"]).size().reset_index(name="n")
    return df

def create_comorbidity_pop_matrix(
    pop: pd.DataFrame, condition: str, parameters: Parameters
) -> NDArray[Any]:
    """
    Create and return the population matrix as a numpy array for calculating the probability of 
    incidence of any of the 9 comorbidities. Each comorbidity has a unique set of variables as 
    listed below.
    """
    # Calculate some intermediate variables needed for the population matrix
    pop["time_since_art"] = pop["year"] - pop["h1yy"]
    pop["out_care"] = (pop["status"] == ART_NONUSER).astype(int)
    if condition in STAGE2 + STAGE3:
        pop["delta_bmi_"] = restricted_cubic_spline_var(
            pop["delta_bmi"], parameters.delta_bmi_dict[condition], 1
        )
        pop["delta_bmi__"] = restricted_cubic_spline_var(
            pop["delta_bmi"], parameters.delta_bmi_dict[condition], 2
        )
        pop["post_art_bmi_"] = restricted_cubic_spline_var(
            pop["post_art_bmi"], parameters.post_art_bmi_dict[condition], 1
        )
        pop["post_art_bmi__"] = restricted_cubic_spline_var(
            pop["post_art_bmi"], parameters.post_art_bmi_dict[condition], 2
        )

    if condition == "anx":
        return np.array(
            pop[
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
            ],
            dtype=float,
        )
    elif condition == "dpr":
        return np.array(
            pop[
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
            ],
            dtype=float,
        )
    elif condition == "ckd":
        return np.array(
            pop[
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
            ],
            dtype=float,
        )
    elif condition == "lipid":
        return np.array(
            pop[
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
            ],
            dtype=float,
        )
    elif condition == "dm":
        return np.array(
            pop[
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
            ],
            dtype=float,
        )
    elif condition == "ht":
        return np.array(
            pop[
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
            ],
            dtype=float,
        )
    elif condition in ["malig", "esld", "mi"]:
        return np.array(
            pop[
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
            ],
            dtype=float,
        )
    else:
        return np.array([], dtype=float)
