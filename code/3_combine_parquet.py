import os
import yaml
import shutil
import pandas as pd
from pathlib import Path
from dask import dataframe as dd
from dask import delayed
from datetime import datetime
import argparse

sa_types = ['type1', 'type2', 'aim2_inc', 'aim2_prev', 'aim2_mort']

start_time = datetime.now()

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--in_dir')
parser.add_argument('--out_dir')
args = parser.parse_args()

in_dir = Path(args.in_dir)
if not args.out_dir:
    out_dir = Path(args.in_dir).parent/'combined'
else:
    out_dir = Path(args.out_dir)
    
if out_dir.is_dir(): #creating output folders
    shutil.rmtree(out_dir)
out_dir.mkdir()

# Load config_file
try:
    with open(in_dir/'../config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    config = {'sa_type' : None}
if config['sa_type'] in sa_types:
    model_names = next(os.walk(in_dir))[1]
    group_names = next(os.walk(in_dir/model_names[0]))[1]
    replications = next(os.walk(in_dir / model_names[0] / group_names[0]))[1]
    output_tables = [x for x in next(os.walk(in_dir/model_names[0]/group_names[0]/replications[0]))[2] if x != 'random.state']
else:
    model_names = ['base']
    group_names = next(os.walk(in_dir))[1]
    replications = next(os.walk(in_dir/group_names[0]))[1]
    output_tables = [x for x in next(os.walk(in_dir/group_names[0]/replications[0]))[2] if x != 'random.state']

test = ['bmi_int_dm_prev', 'cd4_in_care']

for output_table in output_tables:
    if Path(output_table).stem not in test:
        continue
    table_start = datetime.now()
    chunk_list = []
    for model_name in model_names:
        for group_name in group_names:
            for replication in replications:
                replication_int = int(replication.split(sep='_')[1])
                chunk_list.append(in_dir/group_name/replication/output_table)
    df = dd.read_parquet(chunk_list).assign(model=model_name,
                                            group=group_name,
                                            replication=replication_int)
    df = df.astype({'model' : 'category',
                    'group' : 'category',
                    'replication' : 'int16'})
    df = df.repartition(partition_size="100MB")
    df.to_parquet(out_dir/f'{Path(output_table).stem}.parquet')
    table_end = datetime.now()
    print(f'Table {output_table} took: {table_end - table_start}')

# Copy the config file to the output directory
shutil.copy(in_dir/'../config.yaml', out_dir/'config.yaml')

end_time = datetime.now()
print(f'Elapsed Time: {end_time - start_time}')
