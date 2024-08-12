"""
Script to run aggregators, this will be replaced by snakemake
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil

from dask import dataframe as dd

from pearl.post_processing.aggregators import bmi_info

if __name__ == "__main__":
    start_time = datetime.now()

    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    parquet_path = in_dir / "*/*/population.parquet"
    if not args.out_dir:
        out_dir = Path(args.in_dir).parent / "combined"
    else:
        out_dir = Path(args.out_dir)

    if out_dir.is_dir():  # creating output folders
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    population_df = dd.read_parquet(parquet_path)

    bmi_info(population_df, out_dir)

    end_time = datetime.now()
    print(f"Elapsed Time: {end_time - start_time}")
