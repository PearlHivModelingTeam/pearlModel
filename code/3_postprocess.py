# Imports
import os
import shutil
import platform
import ray
import pearl
import yaml
import pkg_resources
import subprocess
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

pd.set_option('io.hdf.default_format', 'table')
start_time = datetime.now()

measured_vars_dict = {'in_care_age': 'n',
                      'out_care_age': 'n',
                      'reengaged_age': 'n',
                      'ltfu_age': 'n',
                      'dead_in_care_age': 'n',
                      'dead_out_care_age': 'n',
                      'new_init_age': 'n',
                      'years_out': 'n',
                      'cd4_inits': 'n',
                      'cd4_inits_2009': 'n',
                      'cd4_in_care': 'n',
                      'cd4_out_care': 'n',
                      'art_coeffs': 'estimate'}

uncombinable_tables = ['art_coeffs']
in_dir = Path('../out/all_aim2_2022-03-23')
store_path = in_dir/'simulation_data.h5'
if store_path.is_file():
    os.remove(store_path)

group_names = next(os.walk(in_dir))[1]
replications = next(os.walk(in_dir/group_names[0]))[1]
output_tables = next(os.walk(in_dir/group_names[0]/replications[0]))[2]
output_tables.remove('random.state')
output_tables = [os.path.splitext(output_table)[0] for output_table in output_tables]
print(output_tables)
output_tables = [output_tables[13]]

with pd.HDFStore(in_dir/'simulation_data.h5') as store:
    for output_table in output_tables:
        print(output_table)
        chunk_list = []
        for group_name in group_names:
            for replication in replications[:20]:
                chunk_list.append(pd.read_csv(in_dir/group_name/replication/f'{output_table}.csv'))
        df = pd.concat(chunk_list, ignore_index=True)
        print(df)
        measured_var = measured_vars_dict[output_table]
        if 'year' in df.columns:
            year_var = 'year'
        elif 'h1yy' in df.columns:
            year_var = 'h1yy'
        if output_table not in uncombinable_tables:
            groupby_cols = list(df.columns.drop(['group', measured_var]))
            ov = df.groupby(groupby_cols)[measured_var].sum().reset_index().assign(group='overall')
            df = pd.concat([df, ov], ignore_index=True)
        table_specific_cols = list(set(df.columns) - set(['group', 'replication', year_var, measured_var]))
        df = df.set_index(['group', 'replication', year_var] + table_specific_cols + [measured_var]).sort_index().reset_index()
        store[output_table] = df
        print(df)
