# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


print(prop_ltfu)
print(n_out_2010_2015)

"""
for name, grouped in in_care_count.groupby(['group']): 
    fig = plt.figure()
    means = grouped.reset_index().groupby(['age_cat', 'year']).mean().drop(columns='replication').rename(columns={'n': 'mean'})
    total = means.reset_index().sort_values('year').set_index(['year', 'age_cat'])
    total = total.groupby('year').sum().rename(columns={'mean':'total'})

    for age_cat, df in means.groupby(level=0):
        means.loc[age_cat] = 100.0 * df / total.values

    means = means.reset_index().sort_values(by='age_cat', ascending=False)

    sns.set(style='darkgrid')

    # Plot data
    ax = sns.FacetGrid(means, col='age_cat', col_order = means.age_cat.unique(), col_wrap=3)
    ax.map(plt.scatter, 'year', 'mean').set_axis_labels('Year', 'Percent')

    # Make title
    plt.subplots_adjust(top=0.9)
    ax.fig.suptitle(name)

    plt.savefig(out_dir + '/fig/' + name + '.png', bbox_inches='tight')
    plt.close(fig)
"""
