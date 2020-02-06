import os
import numpy as np
import pandas as pd
import feather
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", 1001)

# Define directories
cwd = os.getcwd()
param_dir = cwd + '/../../data/parameters'
fig_dir   = os.getcwd() + '/../../out/fig/new_dx_reduce'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

new_dx_interval = feather.read_dataframe(f'{param_dir}/new_dx_interval.feather')

dx_2020 = new_dx_interval.loc[new_dx_interval['year'] == 2020].copy()
dx_2020['2020'] = (dx_2020['upper'] + dx_2020['lower']) / 2.0
dx_2020['diff'] = (dx_2020['upper'] - dx_2020['lower']) / 2.0
dx_reduce = dx_2020[['group', '2020']].set_index('group')

dx_reduce['2021'] = dx_reduce['2020'] - 0.1875 * dx_reduce['2020']
dx_reduce['2022'] = dx_reduce['2021'] - 0.1875 * dx_reduce['2020']
dx_reduce['2023'] = dx_reduce['2022'] - 0.1875 * dx_reduce['2020']
dx_reduce['2024'] = dx_reduce['2023'] - 0.1875 * dx_reduce['2020']
dx_reduce['2025'] = dx_reduce['2024']
dx_reduce['2026'] = dx_reduce['2024']
dx_reduce['2027'] = dx_reduce['2024']
dx_reduce['2028'] = dx_reduce['2024']
dx_reduce['2029'] = dx_reduce['2024']
dx_reduce['2030'] = dx_reduce['2024']
dx_reduce = dx_reduce.reset_index()

year_list = [str(year) for year in range(2020, 2031)]


dx_reduce = pd.melt(dx_reduce, id_vars=['group'], value_vars=year_list, var_name='year', value_name='n').sort_values(['group', 'year'])

dx_diff = dx_2020['diff'].to_numpy()
total_years = len(year_list)
dx_reduce['diff'] = np.array([total_years*[diff] for diff in dx_diff]).flatten()

dx_reduce['lower'] = dx_reduce['n'] - dx_reduce['diff']
dx_reduce['upper'] = dx_reduce['n'] + dx_reduce['diff']
dx_reduce = dx_reduce[['group', 'year', 'lower', 'upper']]
dx_reduce['lower'][dx_reduce['lower'] < 0] = 1.0
#dx_reduce = dx_reduce.set_index(['group', 'year'])

new_dx_interval = new_dx_interval.loc[new_dx_interval['year'] < 2020]

new_dx_interval_reduce = pd.concat([dx_reduce, new_dx_interval]).sort_values(['group', 'year'])
new_dx_interval_reduce['year'] = new_dx_interval_reduce['year'].astype(int)
new_dx_interval_reduce = new_dx_interval_reduce.reset_index(drop=True)
new_dx_interval_reduce.to_feather(f'{param_dir}/new_dx_interval_reduce.feather')

#plots = {'msm': ['msm_black_male', 'msm_hisp_male', 'msm_white_male'],
#         'het_female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
#         'het_male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
#         'idu_female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
#         'idu_male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male']}
#
##group_names = ['msm_black_male', 'msm_hisp_male', 'msm_white_male']
#titles = ['Black', 'Hispanic', 'White']
#
#new_dx = feather.read_dataframe(f'{param_dir}/new_dx.feather')
#new_dx_int = new_dx_interval_reduce.loc[new_dx_interval_reduce['year'] > 2015].copy().set_index(['group', 'year'])
#
#sns.set(style='ticks')
#sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})
#colors = ['navy', 'mediumpurple', 'darkred']
#for plot_name in plots:
#    group_names = plots[plot_name]
#    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(16.0, 9.0))
#
#    for i, (group_name, ax) in enumerate(zip(group_names, axes.flat)):
#        data = new_dx_int.loc[group_name].copy().reset_index()
#        print(data)
#        ax.plot(data['year'], data['lower'], color='b', label='Predicted')
#        ax.plot(data['year'], data['upper'], color='b')
#        ax.fill_between(data['year'], data['lower'], data['upper'], color='b', alpha=0.3)
#
#        na_df = new_dx.loc[new_dx['group'] == group_name].copy()
#        ax.plot(na_df['year'], na_df['n_dx'], 'o', color='k', label='CDC')
#
#        ax.set_title(titles[i])
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
#        if i==0:
#            ax.set_ylabel('Number of New Diagnoses')
#        if i==1:
#            ax.set_xlabel('Year')
#        if i==2:
#            ax.legend(frameon=False, loc='upper left')
#
#
#    plt.suptitle(plot_name)
#    plt.savefig(f'{fig_dir}/{plot_name}.png', bbox_inches='tight')


