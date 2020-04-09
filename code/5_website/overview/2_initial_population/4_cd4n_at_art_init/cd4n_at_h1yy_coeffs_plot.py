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
titles = ['Black', 'Hispanic', 'White']

params = [r'Mean of $\sqrt{\mathrm{CD4}}$, $\mu$', r'Standard Deviation of $\sqrt{\mathrm{CD4}}$, $\sigma$']

risk_lists = [['het_black_female', 'het_hisp_female', 'het_white_female'],
              ['het_black_male', 'het_hisp_male', 'het_white_male'],
              ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
              ['idu_black_male', 'idu_hisp_male', 'idu_white_male'],
              ['msm_black_male', 'msm_hisp_male', 'msm_white_male']]

with pd.HDFStore(param_dir + '/parameters.h5') as store:
    data = store['cd4n_by_h1yy_2009']

data_raw = feather.read_dataframe(f'{param_dir}/aim1/cd4n_by_h1yy_2009_raw.feather')

years = np.arange(2000, 2010)

sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3.0})
colors = ['#7fcdbb', '#2c7fb8']

for risk, risk_list in zip(risks, risk_lists):
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey='row', figsize=(1.2*16, 0.9*16))
    for row, (cols, param) in enumerate(zip(axes, params)):
        for col, (ax, group) in enumerate(zip(cols, risk_list)):
            df = data.loc[group].copy()
            df_raw = data_raw.loc[data_raw['group']==group]
            if row==0:
                ax.set_title(titles[col])
                ax.scatter(df_raw['H1YY'], df_raw['sqrtcd4n_mean'], color='k', label='NA-ACCORD')
                ax.plot(years, df['meanint'] + df['meanslp'] * years, color = colors[row], label='Predicted $\mu$')
                #ax.set_ylim(15.0, 30.0)
            if row==1:
                ax.scatter(df_raw['H1YY'], df_raw['sqrtcd4n_sd'], color='k', label='NA-ACCORD')
                ax.plot(years, df['stdint'] + df['stdslp'] * years, color = colors[row], label='Predicted $\sigma$')
                #ax.set_ylim(5.0, 15.0)
                if col==1:
                    ax.set_xlabel('ART Initiation Year')
            if col==0:
                ax.set_ylabel(params[row])
            if col==2:
                ax.legend(frameon=False)
            ax.tick_params(left=True, labelleft=True)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    plt.savefig(f'out/{risk}.png', bbox_inches='tight')
