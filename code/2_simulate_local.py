# similar version of the code to run simulations locally
# python3 2_simulate_local.py --config="test.yaml"

# Imports
import shutil
import platform
import yaml
import pkg_resources
import subprocess
from pathlib import Path
import argparse
from datetime import datetime
import pearl
import sys
print("Running Python version=" , sys.version)
#Running Python version= 3.6.0 (v3.6.0:41df79263a11, Dec 22 2016, 17:23:13) #output from PK Mac

def run(group_name_run, replication_run, output_path):
    replication_run_str = str(replication_run).zfill(len(str(config['replications'])))
    output_path = output_path
    rerun_path = rerun_root_path/'csv_output'/group_name_run/f'replication_{replication_run_str}' if rerun_root_path is not None else None
    parameters = pearl.Parameters(path=param_file_path, rerun_folder=rerun_path, output_folder=output_path,
                                  group_name=group_name_run, comorbidity_flag=config['comorbidity_flag'], new_dx=config['new_dx'],
                                  final_year=config['final_year'], mortality_model=config['mortality_model'],
                                  mortality_threshold_flag=config['mortality_threshold_flag'], idu_threshold=config['idu_threshold'],
                                  verbose=config['verbose'], bmi_intervention=config['bmi_intervention'],
                                  bmi_intervention_coverage=config['bmi_intervention_coverage'],bmi_intervention_effectiveness=config['bmi_intervention_effectiveness'])
    pearl.Pearl(parameters, group_name_run, replication_run)


def run_sa(sa_variable_run, sa_value_run, group_name_run, replication_run):
    replication_run_str = str(replication_run).zfill(len(str(config['replications'])))
    output_path = output_root_path/'csv_output'/f'{sa_variable_run}_{sa_value_run}'/group_name_run/f'replication_{replication_run_str}'
    rerun_path = rerun_root_path/'csv_output'/f'{sa_variable_run}_{sa_value_run}'/group_name_run/f'replication_{replication_run_str}' if rerun_root_path is not None else None
    parameters = pearl.Parameters(path=param_file_path, rerun_folder=rerun_path, output_folder=output_path,
                                  group_name=group_name_run, comorbidity_flag=config['comorbidity_flag'], new_dx=config['new_dx'],
                                  final_year=config['final_year'], mortality_model=config['mortality_model'],
                                  mortality_threshold_flag=config['mortality_threshold_flag'], idu_threshold=config['idu_threshold'],
                                  verbose=config['verbose'], sa_type=config['sa_type'], sa_variable=sa_variable_run,
                                  sa_value=sa_value_run, bmi_intervention=config['bmi_intervention'],
                                  bmi_intervention_coverage=config['bmi_intervention_coverage'],bmi_intervention_effectiveness=config['bmi_intervention_effectiveness'])
    pearl.Pearl(parameters, group_name_run, replication_run)


start_time = datetime.now()
# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--rerun') #what is a rerun?
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

pearl_path = Path('..')
date_string = datetime.today().strftime('%Y-%m-%d')
param_file_path = pearl_path/'param_files/parameters.h5'

# Check that requirements.txt is met
with open(pearl_path/'requirements.txt', 'r') as requirements:
    pkg_resources.require(requirements)

# Original run, rerun, or test run?
rerun_root_path = None
if args.config:
    config_file_path = pearl_path/'config'/args.config
    output_root_path = pearl_path/f'out/{config_file_path.stem}_{date_string}'
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

# set the output path:
print(f'output directory set to {output_root_path}')

# If it's a rerun check that python version and commit hash are correct else save those details for future runs
if args.rerun:
    print('This is a rerun')
    python_version = config['python_version']
    if python_version != platform.python_version():
        raise EnvironmentError("Incorrect python version for rerun")
    commit_hash = config['commit_hash']
    if commit_hash != subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip():
        raise EnvironmentError("Incorrect commit hash for rerun")
else:
    config['python_version'] = platform.python_version()
    config['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# Load sensitivity analysis variables
if config['sa_type'] == 'none':
    sa_variables = None
    sa_values = None
elif config['sa_type'] == 'type1':
    sa_variables = pearl.sa_type1_var
    sa_values = ['low', 'high']
elif config['sa_type'] == 'type2':
    sa_variables = pearl.sa_type2_var
    sa_values = [0.8, 1.2]
elif (config['sa_type'] == 'aim2_inc') | (config['sa_type'] == 'aim2_prev'):
    sa_variables = pearl.sa_aim2_var
    sa_values = [0.75, 1.25]
elif config['sa_type'] == 'aim2_mort':
    sa_variables = pearl.sa_aim2_mort_var
    sa_values = [0.75, 1.25]
else:
    raise ValueError("Unrecognized sensitivity analysis type")

# Create Output folder structure
if output_root_path.is_dir():
    if (config_file_path.stem == 'test') | args.overwrite:
        shutil.rmtree(output_root_path)
    else:
        raise FileExistsError("Output folder already exists")

if sa_variables is None:
    for group_name in config['group_names']:
        for replication in range(config['replications']):
            replication_str = str(replication).zfill(len(str(config['replications'])))
            output_path = output_root_path/'csv_output'/group_name/f"bmi_{config['bmi_intervention']}/replication_{replication_str}"
            if not output_path.is_dir():  # Check if the directory already exists
                output_path.mkdir(parents=True)
else:
    for sa_variable in sa_variables:
        for sa_value in sa_values:
            for group_name in config['group_names']:
                for replication in range(config['replications']):
                    replication_str = str(replication).zfill(len(str(config['replications'])))
                    output_path = output_root_path/'csv_output'/f'{sa_variable}_{sa_value}'/group_name/f'replication_{replication_str}'
                    if not output_path.is_dir():  # Check if the directory already exists
                        output_path.mkdir(parents=True)

# Copy config file to output dir
with open(output_root_path/'config.yaml', 'w') as yaml_file:
    yaml.safe_dump(config, yaml_file)
########################################################################
# Initialize locally (set up only for the main analysis so far)
if sa_variables is None:
    for group_name in config['group_names']:
        for replication in range(config['replications']):
            replication_str = str(replication).zfill(len(str(config['replications'])))
            output_path = output_root_path/'csv_output'/group_name/f"bmi_{config['bmi_intervention']}/replication_{replication_str}"
            print(f'output path is set to:    {output_path}')
            if not output_path.is_dir():  # Check if the directory already exists
                output_path.mkdir(parents=True)
            print(f'Running the model for {group_name}, replication {replication}')
            run(group_name, replication, output_path)  # Execute the task
else:
    for sa_variable in sa_variables:
        for sa_value in sa_values:
            for group_name in config['group_names']:
                for replication in range(config['replications']):
                    run_sa(sa_variable, sa_value, group_name, replication)

end_time = datetime.now()
print(f'Elapsed Time: {end_time - start_time}')
