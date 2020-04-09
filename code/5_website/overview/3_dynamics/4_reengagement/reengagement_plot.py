# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import feather
import statsmodels.api as sm
from scipy.stats import nbinom
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define directories
cwd = os.getcwd()
input_dir = cwd + '/../../../../../data/input/aim1'
param_dir = cwd + '/../../../../../data/parameters'

obs = pd.read_csv(f'{input_dir}/time_out_naaccord.csv')
obs['p'] = obs['n'] / np.sum(obs['n'])

with pd.HDFStore(param_dir + '/parameters.h5') as store:
    data = store['years_out_of_care']

sns.set(style='ticks')
sns.set_context('paper', font_scale = 1.8, rc={'lines.linewidth':3})

fig, ax = plt.subplots(figsize=(16.0, 9.0))
ax.plot(data['years'], data['probability'], label='Fit')
ax.scatter(obs['years'], obs['p'], label='NA-ACCORD', color ='k')
ax.legend(frameon=False)
ax.set_ylabel('Probability')
ax.set_xlabel('Years Spent Out of Care')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.savefig(f'out/years_out_of_care.png', bbox_inches='tight')


