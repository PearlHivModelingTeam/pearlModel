# Imports
import os
import numpy as np
import pandas as pd
import feather
import statsmodels.formula.api as sm

# Define directories
cwd = os.getcwd()
param_dir = cwd + '/../../data/parameters/aim_1'

# Proportion linking to care
years = np.arange(2006, 2031)
linkage_to_care = pd.DataFrame({'year': years})
linkage_obs = pd.DataFrame({'year':[2010, 2011, 2012, 2013, 2014, 2015, 2016], 'proportion': [0.702, 0.704, 0.714, 0.726, 0.745, 0.75, 0.759]})
model = sm.ols('proportion ~ year', data=linkage_obs)
results = model.fit()
linkage_to_care['link_prop'] = results.predict(linkage_to_care).to_numpy()
linkage_to_care = linkage_to_care.set_index('year')

# Proportion beginning ART
art_prop = len(np.arange(2006, 2011)) * [0.7] + [0.85] + len(np.arange(2012, 2031)) * [1.0]

linkage_to_care['art_prop'] = art_prop
linkage_to_care = linkage_to_care.reset_index()

linkage_to_care.to_feather(f'{param_dir}/linkage_to_care.feather')

# Number of Diagnoses
new_dx = feather.read_dataframe(f'{param_dir}/new_dx.feather').set_index(['group', 'year'])
new_dx['lower'] = new_dx['n_dx']
new_dx['upper'] = new_dx['n_dx']
new_dx = new_dx[['lower', 'upper']].reset_index()
new_dx_interval = feather.read_dataframe(f'{param_dir}/new_dx_interval.feather')

new_dx_interval = new_dx_interval.loc[~new_dx_interval['year'].isin(new_dx['year'].unique())]
new_dx = pd.concat([new_dx_interval, new_dx]).reset_index(drop=True)

new_dx.to_feather(f'{param_dir}/new_dx_combined.feather')

