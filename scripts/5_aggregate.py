import argparse
from datetime import datetime
from pathlib import Path

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
    
    # CORRECTED: Point to the diabetes final output file which contains t_dm
    parquet_path = in_dir / "*/*/dm_final_output.parquet"
    
    out_dir = Path(args.in_dir).parent / "combined" if not args.out_dir else Path(args.out_dir)

    # Dask natively handles reading across the nested directory globs
    population_df = dd.read_parquet(parquet_path)

    bmi_info(population_df, out_dir)

    end_time = datetime.now()
    print(f"Elapsed Time: {end_time - start_time}")
