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
in_dir = cwd + '/../../out/processed/calibrate'
param_dir = cwd + '/../../data/parameters/aim_1'
out_dir = cwd + '/out'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
#group_names = group_names[:1]
group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male']

plots = {'msm': ['msm_black_male', 'msm_hisp_male', 'msm_white_male'],
         'het_female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
         'het_male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
         'idu_female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
         'idu_male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male']}

plots = {'msm': ['msm_black_male', 'msm_hisp_male', 'msm_white_male']}
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
for folder_name, prob in zip(folder_names, probs):
    for group_name in group_names:
        data = data.append(feather.read_dataframe(f'{in_dir}/{folder_name}/{group_name}_n_unique_out_care.feather').assign(prob=prob), ignore_index=True)
        data2 = data2.append(feather.read_dataframe(f'{in_dir}/{folder_name}/{group_name}_dead_out_care_count.feather').assign(prob=prob), ignore_index=True)


data = data.set_index(['prob', 'group', 'replication']).sort_index()

data2 = data2.loc[data2['year'].isin(range(2010,2016))].copy()
data2 = pd.DataFrame(data2.groupby(['prob', 'group', 'replication'])['n'].sum())

data['proportion'] = data2['n'] / data['count']

data = data.groupby(['prob', 'group'])['proportion'].mean()
data = data.reset_index()


sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})
output_table = pd.DataFrame()
for plot_name in plots:
    group_names = plots[plot_name]
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(16.0, 9.0))
    for i, (group_name, ax) in enumerate(zip(group_names, axes.flat)):
        df = data.loc[data['group'] == group_name]
        x = df['prob'].values
        y = df['proportion'].values

        popt, pcov = curve_fit(func, x, y)
        ax.scatter(df['prob'], df['proportion'], color='b', label='Simulations')
        x_fit = np.linspace(0.0, 1.0, 1000)
        y_fit = func(x_fit, *popt)
        ax.plot(x_fit, y_fit, label='Fit')
        ref = targets.loc[targets['group'] == group_name]
        y2 = ref['prop'].values
        y2s = np.array(1000 * [y2]).flatten()
        if group_name=='idu_hisp_female':
            ax.axhline(y=y2, color='grey', label='Adjusted Target')
            ax.axhline(y=0.0203, color='k', label='Target')
        elif group_name=='het_white_male':
            ax.axhline(y=y_fit[-1], color='grey', label='Adjusted Target')
            ax.axhline(y=y2, color='k', label='Target')
        else:
            ax.axhline(y=y2, color='k', label='Target')

        if group_name!='het_white_male':
            idx = np.argwhere(np.diff(np.sign(y2s - y_fit))).flatten()[0]
        else:
            idx = -1
        ax.plot(x_fit[idx], y_fit[idx], 'ro', markersize=10, label='Intersection')
        output_table = output_table.append(pd.DataFrame({'group': group_name, 'prob': x_fit[idx]}, index=[0]), ignore_index=True)

        ax.set_title(titles[i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i==0:
            ax.set_ylabel('Proportion Dying Out of Care')
        if i==1:
            ax.set_xlabel('Probability of Reengagement')
        if i==2:
            ax.legend(frameon=False, loc='upper right')


    plt.savefig(f'{out_dir}/{plot_name}.png', bbox_inches='tight')

output_table = output_table.sort_values('group')
output_table.to_csv(f'{param_dir}/prob_reengage.csv', index=False)
print(output_table)
