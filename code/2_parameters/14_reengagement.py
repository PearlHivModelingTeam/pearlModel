# Imports
import os
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define directories
cwd = os.getcwd()
input_dir = f'{os.getcwd()}/../../data/input/aim1'
param_dir = cwd + '/../../data/parameters/aim1'

data = pd.read_csv(f'{input_dir}/time_out_naaccord.csv')

years = data['years'].to_numpy()
years_pred = np.arange(1000)
n = data['n'].to_numpy()
proportion = n / np.sum(n)

def nbinom_1(x, p, C):
    return C *nbinom(1, p).pmf(x)

def decay_exp(x, C, a):
    return C * np.exp(-a * x)

def pois_fit(x, mu, C):
    return C * poisson(mu).pmf(x)

x = years
pois_param, _ = curve_fit(pois_fit, years, proportion)
exp_param, _ = curve_fit(decay_exp, years, proportion)
y = pois_fit(x, *pois_param)
# Normalize
y = y / np.sum(y)
y = y / np.sum(y)
y2 = decay_exp(x, *exp_param)
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
plt.scatter(years, proportion, label='NA-ACCORD', color = 'k')
plt.plot(years, y, label = 'Scaled Poisson', color = colors[1])
plt.plot(years, y2, label = 'Decaying Exponential', color = colors[2])
plt.xlim(0, 8)
plt.xlabel('Years Out Of Care')
plt.ylabel('Probability')
plt.legend(frameon=False)

years_out = pd.DataFrame({'years': x, 'probability': y})

years_out.to_feather(f'{param_dir}/years_out_of_care.feather')






