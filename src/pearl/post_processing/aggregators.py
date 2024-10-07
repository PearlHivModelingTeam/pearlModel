"""
Module for aggregating results from pearl simulation
"""

from pathlib import Path

import pandas as pd


def bmi_info(population_dataframe: pd.DataFrame, out_dir: Path) -> None:
    """_summary_

    Parameters
    ----------
    population_dataframe : pd.DataFrame
        Population DataFrame at any point in the simulation that contains group, replication, and
        t_dm columns.
    out_dir : Path
        Destination path for the parquet file.
    """
    bmi_info_df = (
        population_dataframe.groupby(["group", "replication", "t_dm"])
        .size()
        .reset_index()
        .rename(columns={0: "n"})
    )
    bmi_info_df.to_parquet(out_dir / "bmi_info.parquet")


def bmi_cat(population_dataframe: pd.DataFrame, out_dir: Path) -> None:
    """_summary_

    Parameters
    ----------
    population_dataframe : pd.DataFrame
        Population DataFrame at any point in the simulation that contains group, replication, and
        t_dm columns.
    out_dir : Path
        Destination path for the parquet file.
    """
    # record bmi group at art_initiation
    if "init_bmi_group" in population_dataframe.columns:
        population_dataframe = population_dataframe.drop(columns=["init_bmi_group"])
    population_dataframe = population_dataframe.compute()

    pre_art_bmi_bins = [0, 18.5, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, float("inf")]
    population_dataframe["init_bmi_group"] = pd.cut(
        population_dataframe["pre_art_bmi"], labels=False, bins=pre_art_bmi_bins, right=False
    ).astype("int8")

    dm_final_output = (
        population_dataframe.groupby(
            [
                "group",
                "replication",
                "bmiInt_scenario",
                "h1yy",
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
                "group": "str",
                "replication": "int8",
                "bmiInt_scenario": "int8",
                "h1yy": "int16",
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

    dm_final_output.to_parquet(out_dir / "bmi_cat_final_output.parquet")
    # dm_final_output.to_csv(out_dir / "bmi_cat_final_output.csv")
