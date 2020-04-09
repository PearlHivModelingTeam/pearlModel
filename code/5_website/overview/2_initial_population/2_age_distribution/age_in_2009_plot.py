import os
import feather
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as stats
import seaborn as sns
import statsmodels.stats.weightstats as weightstats
from matplotlib import rc
pd.set_option("display.max_rows", 1001)
rc('text', usetex=True)

# Define directories
cwd = os.getcwd()
param_dir = os.getcwd() + '/../../../../../data/parameters/aim1'
input_dir = os.getcwd() + '/../../../../../data/input/aim1'
out_dir   = os.getcwd() + '/out'

def mixture(coeffs, x):
    if coeffs.loc['lambda1', 'estimate'] > 0.0:
        rv1 = stats.norm(loc=coeffs.loc['mu1', 'estimate'], scale=coeffs.loc['sigma1', 'estimate'])
    else:
        rv1 = None
    rv2 = stats.norm(loc=coeffs.loc['mu2', 'estimate'], scale=coeffs.loc['sigma2', 'estimate'])
    lambda1 = coeffs.loc['lambda1', 'estimate']
    if rv1 is not None:
        y = lambda1 * rv1.pdf(x) + (1.0 - lambda1) * rv2.pdf(x)
    else:
        y = rv2.pdf(x)
    return (y)

ages = np.arange(18, 85, 1.0)

group_dict = {'het_male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
              'het_female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
              'idu_male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male'],
              'idu_female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
              'msm_male': ['msm_black_male', 'msm_hisp_male', 'msm_white_male']}

titles = ['Black', 'Hispanic', 'White']

age_in_2009 = feather.read_dataframe(f'{param_dir}/age_in_2009.feather')[['group', 'term', 'estimate']].copy()
naaccord_2009 = feather.read_dataframe(f'{input_dir}/naaccord_2009.feather')[['group', 'age2009']].copy()

sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})

for key in group_dict:
    group_names = group_dict[key]
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(16.0, 9.0))
    for i, (group_name, ax) in enumerate(zip(group_names, axes.flat)):
        coeffs = age_in_2009.loc[age_in_2009['group'] == group_name].copy()[['term', 'estimate']].set_index('term')
        naaccord = naaccord_2009.loc[naaccord_2009['group'] == group_name].copy()
        naaccord_ages = naaccord['age2009'].values
        n = len(naaccord_ages)
        iq = stats.iqr(naaccord_ages)
        width = 2 * iq / np.power(n, (1/3))
        rng = stats.iqr(naaccord_ages, rng=(0, 100))
        bins = np.floor(rng/width).astype(int)
        if group_name=='het_black_male':
            bins = bins - 5
        if group_name=='het_white_female':
            bins = bins - 5
        if group_name=='msm_white_male':
            bins = bins - 20
        if group_name=='idu_white_male':
            bins = bins - 20

        ax.hist(naaccord['age2009'].values, density=True, bins = bins, label='NA-ACCORD')
        ax.plot(ages, mixture(coeffs, ages), color = 'k', label='Fit')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(titles[i])
        if i==0:
            ax.set_ylabel('Density')
        if i==1:
            ax.set_xlabel('Age')
        if i==2:
            ax.legend(frameon=False)



    plt.ylim((0.0, 0.07))

    plt.savefig(f'{out_dir}/{key}.png', bbox_inches='tight')


