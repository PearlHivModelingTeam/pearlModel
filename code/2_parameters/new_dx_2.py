# Imports
import os
import numpy as np
import pandas as pd
import feather
import statsmodels.formula.api as sm

# Define directories
cwd = os.getcwd()
param_dir = cwd + '/../../data/parameters/'
aim_1_dir = cwd + '/../../data/parameters/aim_1'
aim_2_dir = cwd + '/../../data/parameters/aim_2'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

# Number of Diagnoses
new_dx = feather.read_dataframe(f'{aim_1_dir}/new_dx.feather').set_index(['group', 'year'])
new_dx['lower'] = new_dx['n_dx']
new_dx['upper'] = new_dx['n_dx']
new_dx = new_dx[['lower', 'upper']].reset_index()
new_dx_interval = feather.read_dataframe(f'{aim_1_dir}/new_dx_interval.feather')

new_dx_interval = new_dx_interval.loc[~new_dx_interval['year'].isin(new_dx['year'].unique())]
new_dx = pd.concat([new_dx_interval, new_dx])
new_dx = new_dx.sort_values(['group', 'year']).set_index(['group', 'year'])

# Proportion linking to care
years = new_dx.index.levels[1].unique()
linkage_to_care = pd.DataFrame({'year': years})
linkage_obs = pd.DataFrame({'year':[2010, 2011, 2012, 2013, 2014, 2015, 2016], 'proportion': [0.702, 0.704, 0.714, 0.726, 0.745, 0.75, 0.759]})
model = sm.ols('proportion ~ year', data=linkage_obs)
results = model.fit()
linkage_to_care['proportion'] = results.predict(linkage_to_care).to_numpy()
linkage_to_care = linkage_to_care.set_index('year')

# Proportion beginning ART
prop = len(np.arange(2006, 2011)) * [0.7] + [0.85] + len(np.arange(2012, 2031)) * [1.0]
begin_art = pd.DataFrame({'year': years, 'proportion': prop}).set_index('year')

# Test
test_pop = new_dx.loc['msm_white_male']
test_pop['n_dx'] = test_pop['lower'] + (test_pop['upper'] - test_pop['lower']) * np.random.uniform(size=len(years))
test_pop = test_pop[['n_dx']]
test_pop['linkage_prop'] = linkage_to_care['proportion']
test_pop['unlinked'] = test_pop['n_dx'] * (1 - linkage_to_care['proportion'])
test_pop['gardner_per_year'] = test_pop['unlinked'] * 0.4 / 3.0

test_pop['year0'] = test_pop['n_dx'] * linkage_to_care['proportion']
test_pop['year1'] = test_pop['gardner_per_year'].shift(1, fill_value=0)
test_pop['year2'] = test_pop['gardner_per_year'].shift(2, fill_value=0)
test_pop['year3'] = test_pop['gardner_per_year'].shift(3, fill_value=0)

test_pop['total_linked'] = test_pop['year0'] + test_pop['year1'] + test_pop['year2'] + test_pop['year3']
test_pop['art_proportion'] = begin_art['proportion']
test_pop['art_users'] = (test_pop['total_linked'] * begin_art['proportion']).astype(int)
test_pop['art_nonusers'] = (test_pop['total_linked'] *  (1 - begin_art['proportion'])).astype(int)
test_pop.to_csv('new_agents.csv')




