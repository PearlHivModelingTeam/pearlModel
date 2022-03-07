# Imports
import os
import platform
import ray
import pearl
import yaml
import pkg_resources
import subprocess
from pathlib import Path
import argparse
from datetime import datetime


@ray.remote
def run(parameters, group_name, replication):
    """Initialized a single instance of PEARL with the given Parameters class, group name, and replication."""
    simulation = pearl.Pearl(parameters, group_name, replication)
    return simulation.stats


# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--rerun')
args = parser.parse_args()

pearl_dir = Path('..')
date_string = datetime.today().strftime('%Y-%m-%d')
param_file = pearl_dir/'param_files/parameters.h5'

# Check that requirements.txt is met
with open(pearl_dir/'requirements.txt', 'r') as requirements:
    pkg_resources.require(requirements)

# Original run, rerun, or test run?
rerun_folder = None
if args.config:
    config_file = pearl_dir/'config'/args.config
    output_folder = pearl_dir/f'out/{config_file.stem}_{date_string}'
elif args.rerun:
    rerun_folder = pearl_dir/'out'/args.rerun
    config_file = rerun_folder/'config.yaml'
    output_folder = pearl_dir/f'out/{args.rerun}_rerun_{date_string}'
else:
    config_file = pearl_dir/'config/test.yaml'
    output_folder = pearl_dir/f'out/{config_file.stem}_{date_string}'


# Load config_file
with open(config_file, 'r') as yaml_file:
    config_yaml = yaml.safe_load(yaml_file)

# Get config options from config file
num_cpus = config_yaml['num_cpus']
replications = range(config_yaml['replications'])
group_names = config_yaml['group_names']
comorbidity_flag = config_yaml['comorbidity_flag']
mm_detail_flag = config_yaml['mm_detail_flag']
new_dx = config_yaml['new_dx']
final_year = config_yaml['final_year']
mortality_model = config_yaml['mortality_model']
verbose = config_yaml['verbose']

# If it's a rerun check that python version and commit hash are correct else save those details for future runs
if args.rerun:
    print('This is a rerun')
    python_version = config_yaml['python_version']
    if python_version != platform.python_version():
        raise EnvironmentError("Incorrect python version for rerun")
    commit_hash = config_yaml['commit_hash']
    if commit_hash != subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip():
        raise EnvironmentError("Incorrect commit hash for rerun")
else:
    config_yaml['python_version'] = platform.python_version()
    config_yaml['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# Create Output folder structure
if os.path.isdir(output_folder):
    if config_file.stem != 'test':
        raise FileExistsError("Output folder already exists")
else:
    os.makedirs(output_folder)
    os.makedirs(output_folder/'random_states')

# Copy config file to output dir
with open(output_folder/'config.yaml', 'w') as yaml_file:
    yaml.safe_dump(config_yaml, yaml_file)

# Initialize ray with the desired number of threads
ray.init(num_cpus=num_cpus)
out_list = []
for group_name in group_names:
    print(group_name)
    # Create Parameters class
    parameters = pearl.Parameters(path=param_file, rerun_folder=rerun_folder, output_folder=output_folder, group_name=group_name, comorbidity_flag=comorbidity_flag, mm_detail_flag=mm_detail_flag,
                                  new_dx=new_dx, final_year=final_year, mortality_model=mortality_model, verbose=verbose)
    # Tell Ray to call the run function in parallel
    futures = [run.remote(parameters, group_name, replication) for replication in replications]
    # Append all output for each replication together
    out_list.append(pearl.Statistics(ray.get(futures), comorbidity_flag, mm_detail_flag))
# Append all output for each subpopulation together and save as csv
out = pearl.Statistics(out_list, comorbidity_flag, mm_detail_flag)
out.save(output_folder)
