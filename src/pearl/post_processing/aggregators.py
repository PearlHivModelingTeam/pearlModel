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
    bmi_info_df.to_parquet(out_dir / "bmi_info.parquet", compression="zstd")


def bmi_cat(population_dataframe: pd.DataFrame, out_dir: Path) -> None:
    """Collapse the dm_final_output counts over init_age_group.

    Parameters
    ----------
    population_dataframe : pd.DataFrame
        dm_final_output DataFrame, already binned into ``init_bmi_group`` and carrying an ``n``
        count column.
    out_dir : Path
        Destination path for the parquet file.
    """

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
        )["n"]
        .sum()
        .compute()
    )

    dm_final_output = dm_final_output.reset_index()

    dm_final_output.to_parquet(out_dir / "bmi_cat_final_output.parquet", compression="zstd")
