# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
cwd = os.getcwd()
out_dir = f'{cwd}/out'
in_dir = f'{cwd}/../../../data/input/aim1/'
param_dir = f'{cwd}/../../../data/parameters/aim1/'


titles = ['Black', 'Hispanic', 'White']

new_dx = pd.read_csv(f'{in_dir}/new_dx.csv')
new_dx_int = pd.read_csv('out/new_dx_all.csv').assign(rm=0)
new_dx_int_list = []
for i, year in enumerate(range(2009, 2015)):
    new_dx_int_list.append(pd.read_csv(f'out/new_dx_{year}.csv').assign(rm=i+1))

new_dx_int = pd.concat([new_dx_int] + new_dx_int_list)

new_dx_int.loc[new_dx_int['model'] == 'NS 2', 'model'] = 'Spline'
new_dx_int = new_dx_int.set_index(['group', 'rm', 'model', 'year']).sort_index()

plots = {'MSM': ['msm_black_male', 'msm_hisp_male', 'msm_white_male'],
         'HET Female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
         'HET Male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
         'IDU Female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
         'IDU Male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male']}

file_names = ['msm', 'het_female', 'het_male', 'idu_female', 'idu_male']

colors = {'Poisson': 'navy',
          'Gamma': 'mediumpurple',
          'Spline': 'darkred'}

sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})

for file_name, plot_name in zip(file_names, plots):
    group_names = plots[plot_name]
    fig, axes = plt.subplots(nrows=7, ncols=3, sharey='col', sharex='all', figsize=(16.0, 16.0))

    for k, (rm, row) in enumerate(zip(new_dx_int.index.levels[1], axes)):
        for i, (group_name, ax) in enumerate(zip(group_names, row)):
            df = new_dx_int.loc[(group_name, rm)].copy()
            models = ['Poisson', 'Gamma', 'Spline']

            for j, model in enumerate(models):
                data = df.loc[model].reset_index()
                neg = data.loc[data['pred.fit'] <= 0]
                keep = data.set_index('year').copy()
                if (group_name != 'idu_white_female') | (model != 'Gamma'):
                    ax.plot(data['year'], data['pred.fit'], label=model, color=colors[model])
                    ax.fill_between(data['year'], data['lower'], data['upper'], color=colors[model], alpha=0.3)
                if keep.loc[2030, 'pred.fit'] <  1.5 * keep.loc[2020, 'pred.fit']:
                    if keep.loc[2006, 'pred.fit'] < 1.5 * keep.loc[2009, 'pred.fit']:
                        pass
                    else:
                        print(group_name)
                        print(model)
                else:
                    print(group_name)
                    print(model)

            na_df = new_dx.loc[new_dx['group'] == group_name].copy()
            ax.plot(na_df['year'].to_numpy()[rm:], na_df['n_dx'].to_numpy()[rm:], 'o', color='k', label='CDC')

            if k==0:
                ax.set_title(f'{titles[i]} {plot_name}', pad=15)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if (k==3) & (i==0):
                ax.set_ylabel('Number of New Diagnoses', labelpad=15)
            if (k==6) & (i==1):
                ax.set_xlabel('Year', labelpad=15)

    plt.savefig(f'{out_dir}/{file_name}.png', bbox_inches='tight')

no_gam = new_dx_int.reset_index().copy()
no_gam = no_gam.loc[(no_gam['group'] != 'idu_white_female') | (no_gam['model'] != 'Gamma')]
data = pd.DataFrame(no_gam.groupby(['group', 'year'])['upper'].max())
data['lower'] = pd.DataFrame(no_gam.groupby(['group', 'year'])['lower'].min())


base_mod = new_dx_int.reset_index().copy()
base_mod = base_mod.loc[(base_mod['group'] != 'idu_white_female') | (base_mod['model'] != 'Gamma')]
base_mod = base_mod.loc[(base_mod['group'] != 'idu_black_female') | (base_mod['model'] != 'Gamma')]
base_mod = base_mod.loc[(base_mod['group'] != 'idu_black_male') | (base_mod['model'] != 'Gamma')]
base_mod = base_mod.loc[(base_mod['group'] != 'idu_white_female') | (base_mod['model'] != 'Spline')]
base_mod = base_mod.loc[(base_mod['group'] != 'idu_white_male') | (base_mod['model'] != 'Spline')]

data_0 = pd.DataFrame(base_mod.loc[base_mod['rm'] == 0].groupby(['group', 'year'])['upper'].max())
data_0['lower'] = pd.DataFrame(base_mod.loc[base_mod['rm'] == 0].groupby(['group', 'year'])['lower'].min())

data_r = data.reset_index().copy()
data_0r = data_0.reset_index().copy()
data_1 = pd.concat([data_0r.loc[data_0r['year'] <= 2017], data_r.loc[data_r['year'] > 2017]]).set_index(['group', 'year']).sort_index()

for file_name, plot_name in zip(file_names, plots):
    group_names = plots[plot_name]
    fig, axes = plt.subplots(nrows=3, ncols=3, sharey='all', sharex='all', figsize=(16.0, 16.0))

    for i, (group_name, ax) in enumerate(zip(group_names, axes[0])):
        df = data_0.loc[group_name].copy()
        ax.fill_between(df.index, df['lower'], df['upper'], color = 'b', label='All Years')
        ax.set_title(titles[i])
        if i==2:
            ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i, (group_name, ax) in enumerate(zip(group_names, axes[1])):
        df = data.loc[group_name].copy()
        ax.fill_between(df.index, df['lower'], df['upper'], color = 'r', label='Full Range')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i==2:
            ax.legend()
        if i==0:
            ax.set_ylabel('Number of new Dx')

    for i, (group_name, ax) in enumerate(zip(group_names, axes[2])):
        df = data_1.loc[group_name].copy()
        ax.fill_between(df.index, df['lower'], df['upper'], color = 'k', label='Final SA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i==2:
            ax.legend()
        if i==1:
            ax.set_xlabel('Year')

    plt.savefig(f'{out_dir}/{file_name}_range.png', bbox_inches='tight')

data_1.reset_index()[['group', 'year', 'lower', 'upper']].to_csv(f'{param_dir}/new_dx_interval_sa.csv', index=False)

