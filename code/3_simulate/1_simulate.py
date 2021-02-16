# Imports
import os
import platform
import ray
import pearl
import yaml
import shutil
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

rerun_folder = None
if args.config:
    config_file = Path(f'config/{args.config}')
    output_folder = Path(f'{pearl_dir}/out/{config_file.stem}_{date_string}/')
elif args.rerun:
    rerun_folder = Path(f'{pearl_dir}/out/{args.rerun}')
    config_file = Path(f'{rerun_folder}/rerun.yaml')
    output_folder = Path(f'{pearl_dir}/out/{args.rerun}_rerun_{date_string}/')
else:
    config_file = Path('config/test.yaml')
    output_folder = Path(f'{pearl_dir}/out/{config_file.stem}_{date_string}/')


# Load config_file
with open(config_file, 'r') as yaml_file:
    config_yaml = yaml.safe_load(yaml_file)

num_cpus = config_yaml['num_cpus']
replications = range(config_yaml['replications'])
group_names = config_yaml['group_names']
sa_dict = config_yaml['sa_dict']
comorbidity_flag = config_yaml['comorbidity_flag']
mm_detail_flag = config_yaml['mm_detail_flag']
smoking_intervention = config_yaml['smoking_intervention']
new_dx = config_yaml['new_dx']
record_tv_cd4 = config_yaml['record_tv_cd4']
verbose = config_yaml['verbose']
if args.rerun:
    print('rerun')
    python_version = config_yaml['python_version']
    if python_version != platform.python_version():
        raise EnvironmentError("Incorrect python version for rerun")
    commit_hash = config_yaml['commit_hash']
    if commit_hash != subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip():
        raise EnvironmentError("Incorrect commit hash for rerun")
else:
    config_yaml['python_version'] = platform.python_version()
    print(config_yaml['python_version'])
    config_yaml['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    print(config_yaml['commit_hash'])


# Delete old files
if os.path.isdir(output_folder):
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
else:
    os.makedirs(output_folder)
    os.makedirs(f'{output_folder}/random_states')

# Copy config file to output dir

with open(f'{output_folder}/rerun.yaml', 'w') as yaml_file:
    yaml.safe_dump(config_yaml, yaml_file)


# Run simulations
ray.init(num_cpus=num_cpus)
out_list = []
for group_name in group_names:
    print(group_name)
    parameters = pearl.Parameters(path=param_file, rerun_folder=rerun_folder, group_name=group_name, replications=replications, comorbidity_flag=comorbidity_flag,
                                  mm_detail_flag=mm_detail_flag, sa_dict=sa_dict, new_dx=new_dx,
                                  output_folder=output_folder, record_tv_cd4=record_tv_cd4, verbose=verbose,
                                  smoking_intervention=smoking_intervention)
    futures = [run.remote(parameters, group_name, replication) for replication in replications]
    out_list.append(pearl.Statistics(ray.get(futures), comorbidity_flag, mm_detail_flag, record_tv_cd4))

out = pearl.Statistics(out_list, comorbidity_flag, mm_detail_flag, record_tv_cd4)
out.save(output_folder)
