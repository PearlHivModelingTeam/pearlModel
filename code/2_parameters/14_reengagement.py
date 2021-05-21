# Imports
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import poisson
from pathlib import Path


def pois_fit(x, mu, C):
    return C * poisson(mu).pmf(x)


# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
input_dir = f'{pearl_dir}/param/raw'
intermediate_dir = f'{pearl_dir}/param/intermediate'
param_dir = f'{pearl_dir}/param/param'

data = pd.read_csv(f'{input_dir}/time_out_naaccord.csv')
years = data['years'].to_numpy()
years_pred = np.arange(1000)
n = data['n'].to_numpy()
proportion = n / np.sum(n)

pois_param, _ = curve_fit(pois_fit, years, proportion)
y = pois_fit(years, *pois_param)
y = y / np.sum(y)
y = y / np.sum(y)

years_out = pd.DataFrame({'years': years, 'probability': y})
years_out.to_csv(f'{param_dir}/years_out_of_care.csv', index=False)






