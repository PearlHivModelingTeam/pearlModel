import os
import feather
import pandas as pd
import statsmodels.stats.weightstats as weightstats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

#pd.set_option("display.max_rows", 7001)
# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../../out/processed/new_reengage'
param_dir = cwd + '/../../data/parameters/aim1'
out_dir = cwd + '/out'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
#group_names = group_names[:1]
#group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male']

plots = {'msm': ['msm_black_male', 'msm_hisp_male', 'msm_white_male'],
         'het_female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
         'het_male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
         'idu_female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
         'idu_male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male']}

#plots = {'msm': ['msm_black_male', 'msm_hisp_male', 'msm_white_male']}
titles = ['Black', 'Hispanic', 'White']

folder_names = ['10', '30', '50', '70', '90']
probs = [0.1, 0.3, 0.5, 0.7, 0.9]
#probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

targets = pd.read_csv('pct_death_out_of_care.csv')
targets.columns = targets.columns.str.lower()
targets['pop2'] = targets['pop2'].str.lower()
targets.loc[targets['sex']==1, 'sex'] = '_male'
targets.loc[targets['sex']==2, 'sex'] = '_female'
targets['group'] = targets['pop2'] + targets['sex']
targets['prop'] = targets['pct']/100
targets = targets[['group', 'prop']]

#rates = rates[:1]


data = pd.DataFrame()
data2 = pd.DataFrame()
for group_name in group_names:
    data = data.append(feather.read_dataframe(f'{in_dir}/{group_name}_n_unique_out_care.feather'), ignore_index=True)
    data2 = data2.append(feather.read_dataframe(f'{in_dir}/{group_name}_dead_out_care_count.feather'), ignore_index=True)


data = data.set_index(['group', 'replication']).sort_index()

data2 = data2.loc[data2['year'].isin(range(2010,2016))].copy()
data2 = pd.DataFrame(data2.groupby(['group', 'replication'])['n'].sum())


data['proportion'] = data2['n'] / data['count']

data = data.groupby(['group'])['proportion'].mean()
data = data.reset_index()

print(data)

sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})
output_table = pd.DataFrame()
for plot_name in plots:
    print(plot_name)
    group_names = plots[plot_name]
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(16.0, 9.0))
    for i, (group_name, ax) in enumerate(zip(group_names, axes.flat)):
        df = data.loc[data['group'] == group_name, 'proportion'].to_numpy()[0]
        ref = targets.loc[targets['group'] == group_name, 'prop'].to_numpy()[0]
        print(ref)
        ax.axhline(y=df, label='Simulated')
        ax.axhline(y=ref, label='Target', color='k')
        print(df)
        print(ref)

        ax.set_title(titles[i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i==0:
            ax.set_ylabel('Proportion Dying Out of Care')
        if i==2:
            ax.legend(frameon=False, loc='upper right')


    plt.savefig(f'{out_dir}/{plot_name}.png', bbox_inches='tight')

