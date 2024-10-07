"""
Script to run aggregators, this will be replaced by snakemake
"""

import argparse
from datetime import datetime
from pathlib import Path

from dask import dataframe as dd

from pearl.post_processing.aggregators import bmi_cat

if __name__ == "__main__":
    start_time = datetime.now()

    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    parquet_path = in_dir / "*/*/final_state.parquet"
    out_dir = Path(args.in_dir).parent / "combined" if not args.out_dir else Path(args.out_dir)

    population_df = dd.read_parquet(parquet_path)
    bmi_cat(population_df, out_dir)

    end_time = datetime.now()
    print(f"Elapsed Time: {end_time - start_time}")
