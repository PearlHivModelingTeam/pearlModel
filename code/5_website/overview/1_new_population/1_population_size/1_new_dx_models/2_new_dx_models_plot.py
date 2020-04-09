# Imports
import os
import numpy as np
import feather
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
cwd = os.getcwd()
out_dir   = f'{cwd}/out'

#group_names = ['msm_black_male', 'msm_hisp_male', 'msm_white_male']
titles = ['Black', 'Hispanic', 'White']

new_dx = feather.read_dataframe(f'out/new_dx.feather')
new_dx_int = feather.read_dataframe(f'out/new_dx_interval.feather')
new_dx_int.loc[new_dx_int['model'] == 'NS 2', 'model'] = 'Spline'
new_dx_int = new_dx_int.set_index(['group', 'model', 'year'])

plots = {'MSM': ['msm_black_male', 'msm_hisp_male', 'msm_white_male'],
         'HET Female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
         'HET Male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
         'IDU Female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
         'IDU Male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male']}

file_names = ['msm', 'het_female', 'het_male', 'idu_female', 'idu_male']


sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})

for file_name, plot_name in zip(file_names, plots):
    group_names = plots[plot_name]
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(16.0, 9.0))

    for i, (group_name, ax) in enumerate(zip(group_names, axes.flat)):
        colors = ['navy', 'mediumpurple', 'darkred']
        df = new_dx_int.loc[group_name].copy()
        models = ['Poisson', 'Gamma', 'Spline']
        if group_name in ['msm_white_male', 'msm_black_male', 'msm_hisp_male']:
            models = models[:-1]
        elif group_name in ['idu_black_male', 'idu_black_female']:
            models = ['Poisson', 'Spline']
            colors = [colors[0], colors[2]]
        for j, model in enumerate(models):
            data = df.loc[model].reset_index()
            neg = data.loc[data['pred.fit'] <= 0]
            if not neg.empty:
                ind = neg.index[0]
                ax.plot(data['year'][:ind], data['pred.fit'][:ind], label=model, color=colors[j])

            else:
                ax.plot(data['year'], data['pred.fit'], label=model, color=colors[j])

            ax.fill_between(data['year'], data['lower'], data['upper'], color=colors[j], alpha=0.3)

        na_df = new_dx.loc[new_dx['group'] == group_name].copy()
        ax.plot(na_df['year'], na_df['n_dx'], 'o', color='k', label='CDC')

        ax.set_title(f'{titles[i]} {plot_name}')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i==0:
            ax.set_ylabel('Number of New Diagnoses')
        if i==1:
            ax.set_xlabel('Year')
        if i==2:
            ax.legend(frameon=False, loc='upper left')


    plt.savefig(f'{out_dir}/{file_name}.png', bbox_inches='tight')



