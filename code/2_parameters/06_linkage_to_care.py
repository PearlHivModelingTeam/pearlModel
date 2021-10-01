# Imports
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
input_dir = f'{pearl_dir}/param/raw'
intermediate_dir = f'{pearl_dir}/param/intermediate'
param_dir = f'{pearl_dir}/param/param'

years = np.arange(2006, 2036)
linkage_obs = pd.read_csv(f'{input_dir}/linkage_to_care.csv')
linkage_obs = pd.melt(linkage_obs, id_vars='group', var_name='year', value_name='link_prob').sort_values(['group', 'year']).reset_index(drop=True)
linkage_obs['year'] = linkage_obs['year'].astype(int)

linkage_list = []
for group, df in linkage_obs.groupby('group'):
    prediction = pd.DataFrame({'group': group,'year': years})
    model = sm.ols('link_prob ~ year', data = df)
    results = model.fit()
    prediction['link_prob'] = results.predict(prediction).to_numpy()
    linkage_list.append(prediction)

linkage_to_care = pd.concat(linkage_list, ignore_index=True)
linkage_to_care.loc[linkage_to_care['link_prob'] > 0.95, 'link_prob'] = 0.95
linkage_to_care['art_prob'] = 0.97
linkage_to_care.loc[linkage_to_care['year'] < 2011, 'art_prob'] = 0.7
linkage_to_care.loc[linkage_to_care['year'] == 2011, 'art_prob'] = 0.85

linkage_to_care.to_csv(f'{param_dir}/linkage_to_care.csv', index=False)

