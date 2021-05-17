# Imports
import os
import pandas as pd
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
input_dir = f'{pearl_dir}/param/raw'
intermediate_dir = f'{pearl_dir}/param/intermediate'
param_dir = f'{pearl_dir}/param/param'

# Read in new diagnosis models
new_dx = pd.read_csv(f'{intermediate_dir}/new_dx.csv')
new_dx_int = pd.read_csv(f'{intermediate_dir}/new_dx_model_all.csv').assign(rm=0)
new_dx_int_list = [new_dx_int]
for i, year in enumerate(range(2009, 2015)):
    new_dx_read = pd.read_csv(f'{intermediate_dir}/new_dx_model_{year}.csv').assign(rm=i+1)
    new_dx_int_list.append(new_dx_read)

# Combine models into single dataframe
new_dx_int = pd.concat(new_dx_int_list)
new_dx_int.loc[new_dx_int['model'] == 'NS 2', 'model'] = 'Spline'
new_dx_int = new_dx_int.set_index(['group', 'rm', 'model', 'year']).sort_index()

# Remove models that don't fit our criteria from the baseline
new_dx_baseline_models = new_dx_int.reset_index().copy()
new_dx_baseline_models = new_dx_baseline_models.loc[~((new_dx_baseline_models['group'] == 'idu_white_female') & (new_dx_baseline_models['model'] == 'Gamma'))]
new_dx_baseline_models = new_dx_baseline_models.loc[~((new_dx_baseline_models['group'] == 'idu_black_female') & (new_dx_baseline_models['model'] == 'Gamma'))]
new_dx_baseline_models = new_dx_baseline_models.loc[~((new_dx_baseline_models['group'] == 'idu_black_male') & (new_dx_baseline_models['model'] == 'Gamma'))]
new_dx_baseline_models = new_dx_baseline_models.loc[~((new_dx_baseline_models['group'] == 'idu_white_female') & (new_dx_baseline_models['model'] == 'Spline'))]
new_dx_baseline_models = new_dx_baseline_models.loc[~((new_dx_baseline_models['group'] == 'idu_white_male') & (new_dx_baseline_models['model'] == 'Spline'))]

# Take the maximum upper and lower bound
new_dx_baseline = pd.DataFrame(new_dx_baseline_models.loc[new_dx_baseline_models['rm'] == 0].groupby(['group', 'year'])['upper'].max())
new_dx_baseline['lower'] = pd.DataFrame(new_dx_baseline_models.loc[new_dx_baseline_models['rm'] == 0].groupby(['group', 'year'])['lower'].min())

# Remove idu_white_female gamma model from sensitivity analysis
new_dx_sa_models = new_dx_int.reset_index().copy()
new_dx_sa_models = new_dx_sa_models.loc[~((new_dx_sa_models['group'] == 'idu_white_female') & (new_dx_sa_models['model'] == 'Gamma'))]
new_dx_sa = pd.DataFrame(new_dx_sa_models.groupby(['group', 'year'])['upper'].max())
new_dx_sa['lower'] = pd.DataFrame(new_dx_sa_models.groupby(['group', 'year'])['lower'].min())

# Create full sensitivity model with baseline untill 2017 and sensitivity analysis afterwards
new_dx_full_model = pd.concat([new_dx_baseline.reset_index().loc[new_dx_baseline.reset_index()['year'] <= 2017],
                               new_dx_sa.reset_index().loc[new_dx_sa.reset_index()['year'] > 2017]]).set_index(['group', 'year']).sort_index()

# Save
new_dx_full_model.reset_index()[['group', 'year', 'lower', 'upper']].to_csv(f'{param_dir}/new_dx_interval_sa.csv', index=False)
