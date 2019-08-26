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

with pd.HDFStore(out_dir + '/pearl_out.h5') as store:
    n_times_lost        = store['n_times_lost']
    dead_in_care_count  = store['dead_in_care_count']
    dead_out_care_count = store['dead_out_care_count']

    new_in_care_count   = store['new_in_care_count']
    new_out_care_count  = store['new_out_care_count']
    in_care_count       = store['in_care_count']
    out_care_count      = store['out_care_count']

    new_init_count      = store['new_init_count']
    in_care_age         = store['in_care_age']
    out_care_age        = store['out_care_age']

    years_out           = store['years_out']
    prop_ltfu           = store['prop_ltfu']
    n_out_2010_2015     = store['n_out_2010_2015']

feather.write_dataframe(n_times_lost, out_dir + '/feather/n_times_lost.feather')
feather.write_dataframe(dead_in_care_count, out_dir + '/feather/dead_in_care_count.feather')
feather.write_dataframe(dead_out_care_count, out_dir + '/feather/dead_out_care_count.feather')

feather.write_dataframe(new_in_care_count, out_dir + '/feather/new_in_care_count.feather')
feather.write_dataframe(new_out_care_count, out_dir + '/feather/new_out_care_count.feather')
feather.write_dataframe(in_care_count, out_dir + '/feather/in_care_count.feather')
feather.write_dataframe(out_care_count, out_dir + '/feather/out_care_count.feather')

feather.write_dataframe(new_init_count, out_dir + '/feather/new_init_count.feather')
feather.write_dataframe(in_care_age, out_dir + '/feather/in_care_age.feather')
feather.write_dataframe(out_care_age, out_dir + '/feather/out_care_age.feather')

feather.write_dataframe(years_out, out_dir + '/feather/years_out.feather')
feather.write_dataframe(prop_ltfu, out_dir + '/feather/prop_ltfu.feather')
feather.write_dataframe(n_out_2010_2015, out_dir + '/feather/n_out_2010_2015.feather')



