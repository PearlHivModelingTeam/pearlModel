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


# Define parallel ray function
@ray.remote
def run(parameters, group_name, replication):
    simulation = pearl.Pearl(parameters, group_name, replication)
    return simulation.stats


# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--rerun')
args = parser.parse_args()

pearl_dir = Path('../..')
date_string = datetime.today().strftime('%Y-%m-%d')
param_file = Path(f'{pearl_dir}/param/parameters.h5')

# Check that requirements.txt is met
with open(f'{pearl_dir}/requirements.txt', 'r') as requirements:
    pkg_resources.require(requirements)

# Original run, rerun, or test run?
rerun_folder = None
if args.config:
    config_file = Path(f'config/{args.config}')
    output_folder = Path(f'{pearl_dir}/out/{config_file.stem}_c_sa_{date_string}/')
elif args.rerun:
    rerun_folder = Path(f'{pearl_dir}/out/{args.rerun}')
    config_file = Path(f'{rerun_folder}/config.yaml')
    output_folder = Path(f'{pearl_dir}/out/{args.rerun}_rerun_{date_string}/')
else:
    config_file = Path('config/test.yaml')
    output_folder = Path(f'{pearl_dir}/out/{config_file.stem}_c_sa_{date_string}/')

# Load config_file
with open(config_file, 'r') as yaml_file:
    config_yaml = yaml.safe_load(yaml_file)

# Get config options from config file
num_cpus = config_yaml['num_cpus']
replications = range(config_yaml['replications'])
group_names = config_yaml['group_names']
classic_sa_dict = pearl.CLASSIC_SA_DICT
comorbidity_flag = config_yaml['comorbidity_flag']
mm_detail_flag = config_yaml['mm_detail_flag']
new_dx = config_yaml['new_dx']
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
    for param in classic_sa_dict.keys():
        for i in [0, 1]:
            output_folder_sa = f'{output_folder}/{param}_{i}'
            os.makedirs(output_folder_sa)
            os.makedirs(f'{output_folder_sa}/random_states')

# Copy config file to output dir
with open(f'{output_folder}/config.yaml', 'w') as yaml_file:
    yaml.safe_dump(config_yaml, yaml_file)

# Run the simulations
ray.init(num_cpus=num_cpus)
for param in classic_sa_dict.keys():
    for i in [0, 1]:
        output_folder_sa = f'{output_folder}/{param}_{i}'
        if rerun_folder is not None:
            rerun_folder_sa = f'{rerun_folder}/{param}_{i}'
        else:
            rerun_folder_sa = None

        classic_sa_dict_run = classic_sa_dict.copy()
        if i == 0:
            classic_sa_dict_run[param] = 0.9
        elif i == 1:
            classic_sa_dict_run[param] = 1.1

        # Run simulations
        out_list = []

        print(f'{param}_{i}')
        for group_name in group_names:
            print(group_name)
            parameters = pearl.Parameters(path=param_file, rerun_folder=rerun_folder_sa, output_folder=output_folder_sa, group_name=group_name, comorbidity_flag=comorbidity_flag,
                                          mm_detail_flag=mm_detail_flag, new_dx=new_dx, verbose=verbose, classic_sa_dict=classic_sa_dict_run)
            futures = [run.remote(parameters, group_name, replication) for replication in replications]
            out_list.append(pearl.Statistics(ray.get(futures), comorbidity_flag, mm_detail_flag))

        out = pearl.Statistics(out_list, comorbidity_flag, mm_detail_flag)
        out.save(output_folder_sa)
