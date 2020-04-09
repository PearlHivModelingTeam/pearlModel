# Imports
import os
import numpy as np
import feather
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
cwd = os.getcwd()
input_dir = cwd + '/../../../../../../data/input/aim1'
param_dir = cwd + '/../../../../../../data/parameters/aim1'
out_dir   = os.getcwd() + '/out'

plots = {'msm': ['msm_black_male', 'msm_hisp_male', 'msm_white_male'],
         'het_female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
         'het_male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
         'idu_female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
         'idu_male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male']}

#group_names = ['msm_black_male', 'msm_hisp_male', 'msm_white_male']
titles = ['Black', 'Hispanic', 'White']

new_dx_int = feather.read_dataframe(f'{param_dir}/new_dx_interval.feather')
new_dx_int = new_dx_int.set_index(['group', 'year'])


sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})
colors = ['navy', 'mediumpurple', 'darkred']
for plot_name in plots:
    group_names = plots[plot_name]
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(16.0, 9.0))

    for i, (group_name, ax) in enumerate(zip(group_names, axes.flat)):
        data = new_dx_int.loc[group_name].copy().reset_index()
        data['0.25'] = data['lower'] + 0.25 * (data['upper'] - data['lower'])
        data['0.5'] = data['lower'] + 0.5 * (data['upper'] - data['lower'])
        data['0.75'] = data['lower'] + 0.75 * (data['upper'] - data['lower'])
        ax.plot(data['year'], data['lower'], color='k', label='Range')
        ax.plot(data['year'], data['upper'], color='k')
        ax.plot(data['year'], data['0.25'], label='r = 0.25')
        ax.plot(data['year'], data['0.5'], label='r = 0.5')
        ax.plot(data['year'], data['0.75'], label='r = 0.75')
        ax.fill_between(data['year'], data['lower'], data['upper'], color='k', alpha=0.3)

        ax.set_title(titles[i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i==0:
            ax.set_ylabel('Number of New Diagnoses')
        if i==1:
            ax.set_xlabel('Year')
        if i==2:
            ax.legend(frameon=False, loc='upper left')


    plt.savefig(f'{out_dir}/{plot_name}.png', bbox_inches='tight')



