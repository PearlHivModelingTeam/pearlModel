import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil

import dask
from dask import dataframe as dd
from dask import delayed


@delayed
def load_and_write(parquet_file_list, target_path):
    df = dd.read_parquet(parquet_file_list)
    df = df.repartition(partition_size="1000MB")
    df.to_parquet(target_path)


if __name__ == "__main__":
    start_time = datetime.now()

    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.in_dir).parent / "combined" if not args.out_dir else Path(args.out_dir)

    group_names = next(os.walk(in_dir))[1]
    replications = next(os.walk(in_dir / group_names[0]))[1]
    output_tables = [
        x
        for x in next(os.walk(in_dir / group_names[0] / replications[0]))[2]
        if x not in ["random.state", "final_state.parquet"]
    ]
    results = []
    for output_table in output_tables:
        chunk_list = []
        for group_name in group_names:
            for replication in replications:
                replication_int = int(replication.split(sep="_")[1])
                chunk_list.append(in_dir / group_name / replication / output_table)

        results.append(load_and_write(chunk_list, out_dir / f"{Path(output_table).stem}.parquet"))

    dask.compute(results, scheduler="processes")
    # Copy the config file to the output directory
    shutil.copy(in_dir / "../config.yaml", out_dir / "config.yaml")

    end_time = datetime.now()
    print(f"Elapsed Time: {end_time - start_time}")
