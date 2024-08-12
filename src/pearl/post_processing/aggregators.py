"""
Module for aggregating results from pearl simulation
"""

from pathlib import Path

import pandas as pd


def bmi_info(population_dataframe: pd.DataFrame, out_dir: Path) -> None:
    bmi_info_df = (
        population_dataframe.groupby(["group", "replication", "t_dm"])
        .size()
        .reset_index()
        .rename(columns={0: "n"})
    )
    bmi_info_df.to_parquet(out_dir / "bmi_info.parquet")
