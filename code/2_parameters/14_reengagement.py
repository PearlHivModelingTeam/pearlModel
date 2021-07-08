# Imports
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.stats import nbinom
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def pois_fit(k, mu):
    return poisson(mu=mu, loc=1).pmf(k)


# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
input_dir = f'{pearl_dir}/param/raw'
intermediate_dir = f'{pearl_dir}/param/intermediate'
param_dir = f'{pearl_dir}/param/param'

data = pd.read_csv(f'{input_dir}/time_out_naaccord.csv')
years = data['years'].to_numpy()
years_pred = np.arange(1, 10)
n = data['n'].to_numpy()
proportion = n / np.sum(n)


pois_param, _ = curve_fit(pois_fit, years, proportion)

y = pois_fit(years, 1*pois_param)
y = y / np.sum(y)

# SA low
y1 = pois_fit(years, 0.9*pois_param)
y1 = y1 / np.sum(y1)

# SA high
y2 = pois_fit(years, 1.1*pois_param)
y2 = y2 / np.sum(y2)

years_out = pd.DataFrame({'years': years, 'probability': y, 'prob_low': y1, 'prob_high': y2})
years_out.to_csv(f'{param_dir}/years_out_of_care.csv', index=False)






