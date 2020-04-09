# Imports
import feather
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

pd.set_option("display.max_rows", 1001)

# Define directories
cwd = os.getcwd()
param_dir = cwd + '/../../../../../data/parameters'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

risks = ['het_female', 'het_male', 'idu_female', 'idu_male', 'msm_male']
risk_lists = [['het_black_female', 'het_hisp_female', 'het_white_female'], ['het_black_male', 'het_hisp_male', 'het_white_male'], ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
                     ['idu_black_male', 'idu_hisp_male', 'idu_white_male'], ['msm_black_male', 'msm_hisp_male', 'msm_white_male']]

params = ['lambda1', 'mu1', 'mu2', 'sigma1', 'sigma2']
param_names = [r'$\lambda_1$', r'$\mu_1$', r'$\mu_2$', r'$\sigma_1$', r'$\sigma_2$']
titles = ['Black', 'Hispanic', 'White']

with pd.HDFStore(param_dir + '/parameters.h5') as store:
    data = store['age_by_h1yy'].reset_index().sort_values(['group', 'param', 'h1yy']).set_index(['group', 'param', 'h1yy'])

data_raw = feather.read_dataframe(f'{param_dir}/aim1/age_by_h1yy_raw.feather').set_index(['group', 'period'])

# Set seaborn style
sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3.0})
colors = [(166.0/255.0, 206.0/255.0, 227.0/255.0), (31.0/255.0, 120/255.0, 180/255.0), (178/255.0, 223/255.0, 138/255.0)]
for risk, risk_list in zip(risks, risk_lists):

    fig, axes = plt.subplots(nrows=5, ncols=3, sharex=False, sharey='row', figsize=(1.2*16, 1.2*16))
    for row, (cols, param) in enumerate(zip(axes, params)):
        for col, (ax, group) in enumerate(zip(cols, risk_list)):
            if row==0:
                ax.set_title(titles[col])
                ax.set_ylim(0.0, 1.0)
            if col==0:
                ax.set_ylabel(param_names[row])
            if (row==4) & (col==1):
                ax.set_xlabel('Year')

            if param in ['mu1', 'mu2']:
                color = colors[1]
            elif param in ['sigma1', 'sigma2']:
                color = colors[2]
            else:
                color = colors[0]
            df_raw = data_raw.loc[group].copy().reset_index()
            df = data.loc[(group, param)].reset_index()
            ax.scatter(df_raw['period'], df_raw[param], color='k', label='NA-ACCORD')
            ax.plot(df['h1yy'], df['low_value'], color=color, label='Predicted')
            ax.plot(df['h1yy'], df['high_value'], color=color)
            ax.fill_between(df['h1yy'], df['low_value'], df['high_value'], color=color, alpha=0.4)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if row==0 and col==2:
                ax.legend(frameon=False)

    plt.savefig(f'out/{risk}.png', bbox_inches='tight')



