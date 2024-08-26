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
