# Imports
import os
import numpy as np
import pandas as pd
import feather

# print more rows
pd.options.display.max_rows = 6000

# Define directories
cwd = os.getcwd()
h5_dir = cwd + '/../../out/py'
feather_dir = cwd + '/../../out/feather/py_no_reset'


group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
#group_names = ['idu_white_female']
replications = 100

for group_name in group_names:
    in_care_count = pd.DataFrame()
    in_care_age = pd.DataFrame()
    out_care_count = pd.DataFrame()
    out_care_age = pd.DataFrame()
    reengaged_count = pd.DataFrame()
    reengaged_age = pd.DataFrame()
    ltfu_count = pd.DataFrame()
    ltfu_age = pd.DataFrame()
    dead_in_care_count = pd.DataFrame()
    dead_in_care_age = pd.DataFrame()
    dead_out_care_count = pd.DataFrame()
    dead_out_care_age = pd.DataFrame()
    new_init_count = pd.DataFrame()
    new_init_age = pd.DataFrame()

    print(group_name)
    for replication in range(replications):
        print(replication)
        with pd.HDFStore(f'{h5_dir}/{group_name}_{replication}.h5') as store:
            in_care_count = in_care_count.append(store['in_care_count'])
            in_care_age = in_care_age.append(store['in_care_age'])

            out_care_count = out_care_count.append(store['out_care_count'])
            out_care_age = out_care_age.append(store['out_care_age'])

            reengaged_count = reengaged_count.append(store['reengaged_count'])
            reengaged_age = reengaged_age.append(store['reengaged_age'])

            ltfu_count = ltfu_count.append(store['ltfu_count'])
            ltfu_age = ltfu_age.append(store['ltfu_age'])

            dead_in_care_count = dead_in_care_count.append(store['dead_in_care_count'])
            dead_in_care_age = dead_in_care_age.append(store['dead_in_care_age'])

            dead_out_care_count = dead_out_care_count.append(store['dead_out_care_count'])
            dead_out_care_age = dead_out_care_age.append(store['dead_out_care_age'])

            new_init_count = new_init_count.append(store['new_init_count'].rename(columns={'h1yy_orig':'h1yy'}))
            new_init_age = new_init_age.append(store['new_init_age'].rename(columns={'h1yy_orig':'h1yy'}))


    dead_in_care_age = dead_in_care_age.loc[dead_in_care_age.age <= 85]
    dead_out_care_age = dead_out_care_age.loc[dead_out_care_age.age <= 85]

    feather.write_dataframe(in_care_count, f'{feather_dir}/{group_name}_in_care_count.feather')
    feather.write_dataframe(in_care_age, f'{feather_dir}/{group_name}_in_care_age.feather')
    
    feather.write_dataframe(out_care_count, f'{feather_dir}/{group_name}_out_care_count.feather')
    feather.write_dataframe(out_care_age, f'{feather_dir}/{group_name}_out_care_age.feather')

    feather.write_dataframe(reengaged_count, f'{feather_dir}/{group_name}_reengaged_count.feather')
    feather.write_dataframe(reengaged_age, f'{feather_dir}/{group_name}_reengaged_age.feather')

    feather.write_dataframe(ltfu_count, f'{feather_dir}/{group_name}_ltfu_count.feather')
    feather.write_dataframe(ltfu_age, f'{feather_dir}/{group_name}_ltfu_age.feather')

    feather.write_dataframe(dead_in_care_count, f'{feather_dir}/{group_name}_dead_in_care_count.feather')
    feather.write_dataframe(dead_in_care_age, f'{feather_dir}/{group_name}_dead_in_care_age.feather')

    feather.write_dataframe(dead_out_care_count, f'{feather_dir}/{group_name}_dead_out_care_count.feather')
    feather.write_dataframe(dead_out_care_age, f'{feather_dir}/{group_name}_dead_out_care_age.feather')

    feather.write_dataframe(new_init_count, f'{feather_dir}/{group_name}_new_init_count.feather')
    feather.write_dataframe(new_init_age, f'{feather_dir}/{group_name}_new_init_age.feather')

    


