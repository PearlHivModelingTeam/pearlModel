# Imports
import os
import ray
import pearl
import yaml
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


# Define parallel ray function
@ray.remote
def run(parameters, group_name, replication):
    simulation = pearl.Pearl(parameters, group_name, replication)
    return simulation.stats


yaml_file = Path('config/cepac_docking.yaml')

date_string = datetime.today().strftime('%Y-%m-%d')
output_folder = Path(f'../../out/{yaml_file.stem}_{date_string}/')

# Load parameters
with open(yaml_file, 'r') as f:
    param_yaml = yaml.load(f, Loader=yaml.FullLoader)
    num_cpus = param_yaml['num_cpus']
    param_file = Path(param_yaml['param_file'])
    replications = range(param_yaml['replications'])
    group_names = param_yaml['group_names']
    sa_dict = param_yaml['sa_dict']
    comorbidity_flag = param_yaml['comorbidity_flag']
    mm_detail_flag = param_yaml['mm_detail_flag']
    new_dx = param_yaml['new_dx']
    record_tv_cd4 = param_yaml['record_tv_cd4']
    verbose = param_yaml['verbose']

dock_mods_list = pd.DataFrame()
for i in replications:
    dock_mods = pd.DataFrame.from_dict({'disengagement': np.random.uniform(0.5, 2.0),
                                         'reengagement': np.random.uniform(0.5, 2.0),
                                         'mortality_in_care': np.random.uniform(0.5, 2.0),
                                         'mortality_out_care': np.random.uniform(0.5, 2.0)},
                                         orient='index', columns=['value']).assign(replication=i).reset_index()
    dock_mods_list = pd.concat([dock_mods_list, dock_mods], ignore_index=True)
dock_mods_list = dock_mods_list.set_index(['replication', 'index'])



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

dock_mods_list.reset_index().to_csv(f'{output_folder}/dock_mods.csv', index=False)

# Run simulations
ray.init(num_cpus=num_cpus)
out_list = []
for group_name in group_names:
    parameters = pearl.Parameters(path=param_file, group_name=group_name, comorbidity_flag=comorbidity_flag,
                                  mm_detail_flag=mm_detail_flag, sa_dict=sa_dict, new_dx=new_dx,
                                  output_folder=output_folder, record_tv_cd4=record_tv_cd4, verbose=verbose,
                                  dock_mods=dock_mods_list)
    futures = [run.remote(parameters, group_name, replication) for replication in replications]
    out_list.append(pearl.Statistics(ray.get(futures), comorbidity_flag=comorbidity_flag, mm_detail_flag=mm_detail_flag, record_tv_cd4=record_tv_cd4))

out = pearl.Statistics(out_list, comorbidity_flag=comorbidity_flag, mm_detail_flag=mm_detail_flag, record_tv_cd4=record_tv_cd4)
out.save(output_folder)

