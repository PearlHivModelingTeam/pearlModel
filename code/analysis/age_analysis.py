# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# print more rows
pd.options.display.max_rows = 150

# Define directories
cwd = os.getcwd()
out_dir = cwd + '/../../out'


with pd.HDFStore(out_dir + '/out.h5') as store:
    age_final = store['age_final']

group = age_final.loc['msm_black_male'].reset_index()

for name, grouped in age_final.groupby(['group']): 
    fig = plt.figure()
    means = grouped.reset_index().groupby(['age_cat', 'year']).mean().drop(columns='dataset').rename(columns={'n': 'mean'})
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


