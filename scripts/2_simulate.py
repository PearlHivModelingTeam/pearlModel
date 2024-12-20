# Imports
import argparse
from datetime import datetime
from pathlib import Path
import random
import shutil

import dask
import yaml

from pearl.definitions import PROJECT_DIR
from pearl.model import Parameters, Pearl


@dask.delayed
def run(group_name_run, replication_run, seed):
    replication_run_str = str(replication_run).zfill(len(str(config["replications"])))
    out_path = f"parquet_output/{group_name_run}/replication_{replication_run_str}"
    output_folder = output_root_path / out_path
    parameters = Parameters(
        path=param_file_path,
        output_folder=output_folder,
        replication=replication_run,
        group_name=group_name_run,
        new_dx=config["new_dx"],
        final_year=config["final_year"],
        mortality_model=config["mortality_model"],
        mortality_threshold_flag=config["mortality_threshold_flag"],
        idu_threshold=config["idu_threshold"],
        bmi_intervention_scenario=config.get("bmi_intervention_scenario", 0),
        bmi_intervention_start_year=config.get("bmi_intervention_start_year", 2012),
        bmi_intervention_end_year=config.get("bmi_intervention_end_year", 2017),
        bmi_intervention_coverage=config.get("bmi_intervention_coverage", 1),
        bmi_intervention_effectiveness=config.get("bmi_intervention_effectiveness", 1),
        seed=seed,
        sa_variables=config.get("sa_variables", []),
    )
    print(
        f"""Initializing group {group_name_run}, rep {replication_run}:
        output set to {parameters.output_folder}""",
        flush=True,
    )
    Pearl(parameters).run()
    print(f"simulation finished for {group_name_run},rep= {replication_run}", flush=True)


if __name__ == "__main__":
    # set a seed for the main thread
    random.seed(42)
    print("1", flush=True)

    start_time = datetime.now()
    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pearl_path = PROJECT_DIR
    param_file_path = pearl_path / "param_files/parameters.h5"

    # Original run, rerun, or test run?
    rerun_root_path = None
    if args.config:
        config_file_path = Path(args.config)
        output_root_path = pearl_path / f"out/{config_file_path.stem}"
        print(output_root_path.resolve(), flush=True)
    else:
        config_file_path = pearl_path / "config/test.yaml"
        output_root_path = pearl_path / f"out/{config_file_path.stem}"
    # Load config_file
    with Path.open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)

    print("2", flush=True)

    # Create Output folder structure
    if output_root_path.is_dir():
        if (config_file_path.stem == "test") | args.overwrite:
            print("rewriting existing folders")
            shutil.rmtree(output_root_path)
        else:
            raise FileExistsError("Output folder already exists")

    print("3", flush=True)
    for group_name_run in config["group_names"]:
        for replication_run in range(config["replications"]):
            replication_run_str = str(replication_run).zfill(len(str(config["replications"])))
            out_path = f"parquet_output/{group_name_run}/replication_{replication_run_str}"
            output_folder = output_root_path / out_path
            output_folder.mkdir(parents=True)

    # Copy config file to output dir
    with Path.open(output_root_path / "config.yaml", "w") as yaml_file:
        yaml.safe_dump(config, yaml_file)

    # Launching simulations
    print("4", flush=True)

    print("running main analysis...", flush=True)
    num_seeds = len(config["group_names"]) * config["replications"]
    seeds = []
    seed = random.randint(1, 100000000)
    results = []
    for group_name_run in config["group_names"]:
        for replication_run in range(config["replications"]):
            while seed in seeds:
                seed = random.randint(1, 100000000)
            results.append(run(group_name_run, replication_run, seed=seed))
            seeds.append(seed)

    if args.debug:
        dask.compute(results, scheduler="processes", num_workers=config["num_cpus"])
    else:
        dask.compute(results, scheduler="processes")

    end_time = datetime.now()
    print(f"**** Elapsed Time: {end_time - start_time} ****")
