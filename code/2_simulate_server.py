# Imports
import shutil
import platform

from dask.distributed import Client
import pearl
import yaml
import pkg_resources
import subprocess
from pathlib import Path
import argparse
from datetime import datetime
import os


def run(group_name_run, replication_run):
    replication_run_str = str(replication_run).zfill(len(str(config['replications'])))
    out_path = f"csv_output/{group_name_run}/replication_{replication_run_str}" #setting up the path name
    output_folder = output_root_path/out_path
    rerun_folder = rerun_root_path/out_path if rerun_root_path is not None else None
    parameters = pearl.Parameters(path=param_file_path,
                                  rerun_folder=rerun_folder,
                                  output_folder=output_folder,
                                  group_name=group_name_run,
                                  comorbidity_flag=config['comorbidity_flag'],
                                  new_dx=config['new_dx'],
                                  final_year=config['final_year'],
                                  mortality_model=config['mortality_model'],
                                  mortality_threshold_flag=config['mortality_threshold_flag'],
                                  idu_threshold=config['idu_threshold'],
                                  verbose=config['verbose'],
                                  bmi_intervention=config['bmi_intervention'],
                                  bmi_intervention_scenario=config['bmi_intervention_scenario'],
                                  bmi_intervention_start_year=config['bmi_intervention_start_year'],
                                  bmi_intervention_end_year=config['bmi_intervention_end_year'],
                                  bmi_intervention_coverage=config['bmi_intervention_coverage'],
                                  bmi_intervention_effectiveness=config['bmi_intervention_effectiveness'])
    print(f'Initializing group {group_name_run}, rep {replication_run}: output set to {parameters.output_folder}', flush=True)
    pearl.Pearl(parameters, group_name_run, replication_run)
    print(f'simulation finished for {group_name_run},rep= {replication_run}', flush=True)
###############################

if __name__ == '__main__':
    num_rep=20
    print("1", flush=True)
    client = Client(n_workers=num_rep)
    print("2", flush=True)

    start_time = datetime.now()
    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--rerun')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    pearl_path = Path('..')
    date_string = datetime.today().strftime('%Y-%m-%d')
    param_file_path = pearl_path/'param_files/parameters.h5'

    # Check that requirements.txt is met
    """with open(pearl_path/'requirements.txt', 'r') as requirements:
        pkg_resources.require(requirements)"""

    # Original run, rerun, or test run?
    rerun_root_path = None
    if args.config:
        config_file_path = pearl_path/'config'/args.config
        output_root_path = pearl_path/f'out/{config_file_path.stem}_{date_string}'
        print(output_root_path.resolve(), flush=True)
    elif args.rerun:
        rerun_root_path = pearl_path/'out'/args.rerun
        config_file_path = rerun_root_path/'config.yaml'
        output_root_path = pearl_path/f'out/{args.rerun}_rerun_{date_string}'
    else:
        config_file_path = pearl_path/'config/test.yaml'
        output_root_path = pearl_path/f'out/{config_file_path.stem}_{date_string}'
    # Load config_file
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    print("3", flush=True)
    # If it's a rerun check that python version and commit hash are correct else save those details for future runs
    """if args.rerun:
        print('This is a rerun')
        python_version = config['python_version']
        if python_version != platform.python_version():
            raise EnvironmentError("Incorrect python version for rerun")
        commit_hash = config['commit_hash']
        if commit_hash != subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip():
            raise EnvironmentError("Incorrect commit hash for rerun")
    else:
        config['python_version'] = platform.python_version()
        config['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()"""

    # Load sensitivity analysis variables
    if config['sa_type'] == 'none':
        sa_variables = None
        sa_values = None

    # Create Output folder structure
    if output_root_path.is_dir():
        if (config_file_path.stem == 'test') | args.overwrite:
            print("rewriting existing folders")
            shutil.rmtree(output_root_path)
        else:
            raise FileExistsError("Output folder already exists")

    print("4", flush=True)
    if sa_variables is None:
        for group_name_run in config['group_names']:
            for replication_run in range(config['replications']):
                replication_run_str = str(replication_run).zfill(len(str(config['replications'])))
                out_path = f"csv_output/{group_name_run}/replication_{replication_run_str}" #setting up the path name
                output_folder = output_root_path/out_path
                output_folder.mkdir(parents=True)

    # Copy config file to output dir
    with open(output_root_path/'config.yaml', 'w') as yaml_file:
        yaml.safe_dump(config, yaml_file)

    # Launching simulations
    print("5", flush=True)

    results = []
    if sa_variables is None:
        print("running main analysis...", flush=True)
        for replication_run in range(num_rep):
            results.append(client.submit(run, group_name_run, replication_run))

    # Gather results back to local computer
    results = client.gather(results)

    end_time = datetime.now()
    print(f'**** Elapsed Time: {end_time - start_time} ****')
