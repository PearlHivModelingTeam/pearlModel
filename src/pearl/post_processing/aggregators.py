"""
Module for aggregating results from pearl simulation
"""

from pathlib import Path
from typing import List

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

    def find_bin_index(x: int, bins: List[float]) -> int:
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                return i
        return len(bins) - 2

    # record bmi group at art_initiation
    if "init_bmi_group" in population_dataframe.columns:
        population_dataframe = population_dataframe.drop(columns=["init_bmi_group"])
    # population_dataframe = population_dataframe.compute()

    # pre_art_bmi_bins = [0,18.5,19,20,21,22,23,24,25,26,27,28,29,30,float("inf")]
    # population_dataframe["init_bmi_group"] = pd.cut(population_dataframe["pre_art_bmi"], labels=False, bins=pre_art_bmi_bins, right=False).astype("int8")

    pre_art_bmi_bins = [0, 18.5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, float("inf")]

    # Assuming `population_dataframe` is a Dask DataFrame
    # Convert the BMI values to categorical groups using map_partitions
    population_dataframe["init_bmi_group"] = population_dataframe["pre_art_bmi"].apply(
        lambda x: find_bin_index(x, pre_art_bmi_bins)
    )

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
        .compute()
    )

    dm_final_output = dm_final_output.reset_index(name="n")

    dm_final_output.to_parquet(out_dir / "bmi_cat_final_output.parquet")
    # dm_final_output.to_csv(out_dir / "bmi_cat_final_output.csv")
