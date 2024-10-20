# Imports
import argparse
from datetime import datetime
from pathlib import Path
import shutil

import dask
import yaml

from pearl.definitions import PROJECT_DIR
from pearl.model import Parameters, Pearl


@dask.delayed
def run(group_name_run, replication_run, seed=None):
    replication_run_str = str(replication_run).zfill(len(str(config["replications"])))
    out_path = f"parquet_output/{group_name_run}/replication_{replication_run_str}"
    output_folder = output_root_path / out_path
    parameters = Parameters(
        path=param_file_path,
        output_folder=output_folder,
        group_name=group_name_run,
        new_dx=config["new_dx"],
        final_year=config["final_year"],
        mortality_model=config["mortality_model"],
        mortality_threshold_flag=config["mortality_threshold_flag"],
        idu_threshold=config["idu_threshold"],
        bmi_intervention_scenario=config["bmi_intervention_scenario"],
        bmi_intervention_start_year=config["bmi_intervention_start_year"],
        bmi_intervention_end_year=config["bmi_intervention_end_year"],
        bmi_intervention_coverage=config["bmi_intervention_coverage"],
        bmi_intervention_effectiveness=config["bmi_intervention_effectiveness"],
        seed=seed,
    )
    print(
        f"""Initializing group {group_name_run}, rep {replication_run}:
        output set to {parameters.output_folder}""",
        flush=True,
    )
    Pearl(parameters, group_name_run, replication_run).run()
    print(f"simulation finished for {group_name_run},rep= {replication_run}", flush=True)


if __name__ == "__main__":
    print("1", flush=True)

    start_time = datetime.now()
    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--rerun")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pearl_path = PROJECT_DIR
    date_string = datetime.today().strftime("%Y-%m-%d")
    param_file_path = pearl_path / "param_files/parameters.h5"

    # Original run, rerun, or test run?
    rerun_root_path = None
    if args.config:
        config_file_path = Path(args.config)
        output_root_path = pearl_path / f"out/{config_file_path.stem}_{date_string}"
        print(output_root_path.resolve(), flush=True)
    elif args.rerun:
        rerun_root_path = pearl_path / "out" / args.rerun
        config_file_path = rerun_root_path / "config.yaml"
        output_root_path = pearl_path / f"out/{args.rerun}_rerun_{date_string}"
    else:
        config_file_path = pearl_path / "config/test.yaml"
        output_root_path = pearl_path / f"out/{config_file_path.stem}_{date_string}"
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
    results = []
    seed = 0
    for group_name_run in config["group_names"]:
        for replication_run in range(config["replications"]):
            results.append(run(group_name_run, replication_run, seed=seed))
            seed += 1

    if args.debug:
        dask.compute(results, scheduler="processes", num_workers=config["num_cpus"])
    else:
        dask.compute(results, scheduler="processes")

    end_time = datetime.now()
    print(f"**** Elapsed Time: {end_time - start_time} ****")
