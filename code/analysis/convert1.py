# Imports
import os
import numpy as np
import pandas as pd
import feather

# print more rows
pd.options.display.max_rows = 1000

# Define directories
cwd = os.getcwd()
out_dir = cwd + '/../../out'
py_dir = out_dir + '/py'


in_care_count = pd.DataFrame()
out_care_count = pd.DataFrame()
new_in_care_count = pd.DataFrame()
new_out_care_count = pd.DataFrame()
dead_in_care_count = pd.DataFrame()
dead_out_care_count = pd.DataFrame()
new_init_count = pd.DataFrame()
n_times_lost = pd.DataFrame()
in_care_age = pd.DataFrame()
out_care_age = pd.DataFrame()



group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
replications = 100 

for group_name in group_names:
    print(group_name)
    for replication in range(replications):
        print(replication)
        with pd.HDFStore('{}/{}_{}.h5'.format(py_dir, group_name, str(replication))) as store:
            in_care_count = in_care_count.append(store['in_care_count'])
            out_care_count = out_care_count.append(store['out_care_count'])

            new_in_care_count = new_in_care_count.append(store['new_in_care_count'])
            new_out_care_count = new_out_care_count.append(store['new_out_care_count'])

            dead_in_care_count = dead_in_care_count.append(store['dead_in_care_count'])
            dead_out_care_count = dead_out_care_count.append(store['dead_out_care_count'])

            new_init_count = new_init_count.append(store['new_init_count'])
            n_times_lost = n_times_lost.append(store['n_times_lost'])

            in_care_age = in_care_age.append(store['in_care_age'])
            out_care_age = out_care_age.append(store['out_care_age'])

            #new_in_care_age = new_in_care_age.append(store['new_in_care_age'])
            #new_out_care_age = new_out_care_age.append(store['new_out_care_age'])

            #dead_in_care_age = dead_in_care_age.append(store['dead_in_care_age'])
            #dead_out_care_age = dead_out_care_age.append(store['dead_out_care_age'])

feather.write_dataframe(in_care_count, out_dir + '/feather/in_care_count.feather')
feather.write_dataframe(out_care_count, out_dir + '/feather/out_care_count.feather')

feather.write_dataframe(new_in_care_count, out_dir + '/feather/new_in_care_count.feather')
feather.write_dataframe(new_out_care_count, out_dir + '/feather/new_out_care_count.feather')

feather.write_dataframe(dead_in_care_count, out_dir + '/feather/dead_in_care_count.feather')
feather.write_dataframe(dead_out_care_count, out_dir + '/feather/dead_out_care_count.feather')

feather.write_dataframe(new_init_count, out_dir + '/feather/new_init_count.feather')
feather.write_dataframe(n_times_lost, out_dir + '/feather/n_times_lost.feather')

feather.write_dataframe(in_care_age, out_dir + '/feather/in_care_age.feather')
feather.write_dataframe(out_care_age, out_dir + '/feather/out_care_age.feather')

#feather.write_dataframe(new_in_care_age, out_dir + '/feather/new_in_care_age.feather')
#feather.write_dataframe(new_out_care_age, out_dir + '/feather/new_out_care_age.feather')

#feather.write_dataframe(dead_in_care_age, out_dir + '/feather/dead_in_care_age.feather')
#feather.write_dataframe(dead_out_care_age, out_dir + '/feather/dead_out_care_age.feather')
