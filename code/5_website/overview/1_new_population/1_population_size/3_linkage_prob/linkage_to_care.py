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

years = np.arange(2006, 2031)
linkage_obs = pd.read_csv(f'{input_dir}/linkage_to_care.csv')
linkage_obs = pd.melt(linkage_obs, id_vars='group', var_name='year', value_name='link_prob').sort_values(['group', 'year']).reset_index(drop=True)
linkage_obs['year'] = linkage_obs['year'].astype(int)

linkage_to_care = pd.read_feather(f'{param_dir}/linkage_to_care.feather')

sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})
colors = ['navy', 'mediumpurple', 'darkred']
for plot_name in plots:
    group_names = plots[plot_name]
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(16.0, 9.0))

    for i, (group_name, ax) in enumerate(zip(group_names, axes.flat)):
        data = linkage_to_care.loc[linkage_to_care['group']==group_name].copy()
        obs = linkage_obs.loc[linkage_obs['group']==group_name].copy()
        ax.scatter(obs['year'], obs['link_prob'], color='k', label='CDC')
        ax.plot(data['year'], data['link_prob'], label='Predicted')
        ax.set_ylim((0.7, 1.0))
        ax.set_title(titles[i])
        if i==0:
            ax.set_ylabel('Probability of Linkage to Care')
        if i==1:
            ax.set_xlabel('Year')
        if i==2:
            ax.legend(frameon=False)


    plt.savefig(f'{out_dir}/{plot_name}.png', bbox_inches='tight')
