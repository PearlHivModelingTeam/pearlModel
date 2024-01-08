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

start_time = datetime.now()
pearl_path = Path('..')
date_string = datetime.today().strftime('%Y-%m-%d')
param_file_path = pearl_path/'param_files/parameters.h5'
config_file_path = pearl_path/'config/test.yaml'
output_root_path = pearl_path/f'out/{config_file_path.stem}_{date_string}'
print(config_file_path.resolve())
# Load config_file
with open(config_file_path, 'r') as config_file:
    config = yaml.safe_load(config_file)
# set the output path:
print(f'output directory set to {output_root_path}')

# Create Output folder structure
if output_root_path.is_dir():
    if (config_file_path.stem == 'test'):
        shutil.rmtree(output_root_path)
    else:
        raise FileExistsError("Output folder already exists")

for group_name in config['group_names']:
    for replication in range(config['replications']):
        replication_str = str(replication).zfill(len(str(config['replications'])))
        output_path = output_root_path/'csv_output'/group_name/f"bmi_{config['bmi_intervention']}/replication_{replication_str}"
        if not output_path.is_dir():  # Check if the directory already exists
            output_path.mkdir(parents=True)
# Copy config file to output dir
with open(output_root_path/'config.yaml', 'w') as yaml_file:
    yaml.safe_dump(config, yaml_file)
########################################################################
# Initialize locally (set up only for the main analysis so far)
group_name = 'idu_black_female'
replication = 1
replication_str = str(replication).zfill(len(str(config['replications'])))
output_path = output_root_path/'csv_output'/group_name/f"bmi_{config['bmi_intervention']}/replication_{replication_str}"
if not output_path.is_dir():  # Check if the directory already exists
    output_path.mkdir(parents=True)
rerun_path = None
parameters = pearl.Parameters(path=param_file_path,
                              rerun_folder=rerun_path,
                              output_folder=output_path,
                              group_name=group_name,
                              comorbidity_flag=config['comorbidity_flag'],
                              new_dx=config['new_dx'],
                              final_year=config['final_year'],
                              mortality_model=config['mortality_model'],
                              mortality_threshold_flag=config['mortality_threshold_flag'],
                              idu_threshold=config['idu_threshold'],
                              verbose=config['verbose'],
                              bmi_intervention_scenario=config['bmi_intervention_scenario'],
                              bmi_intervention_start_year=config['bmi_intervention_start_year'],
                              bmi_intervention_end_year=config['bmi_intervention_end_year'],
                              bmi_intervention_coverage=config['bmi_intervention_coverage'],
                              bmi_intervention_effectiveness=config['bmi_intervention_effectiveness'])
print(f"Ready to run {group_name}, replication {replication}, to year {config['final_year']}")

pearl.Pearl(parameters, group_name, replication)
