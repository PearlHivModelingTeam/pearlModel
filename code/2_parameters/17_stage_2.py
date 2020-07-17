# Imports
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 1001)

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../../data/input/aim2/stage2'
out_dir = cwd + '/../../data/parameters/aim2/stage2'

# Load ckd coeff csv file
ckd_coeff = pd.read_csv(f'{in_dir}/ckd_coeff.csv')

ckd_coeff.columns = ckd_coeff.columns.str.lower()
ckd_coeff = ckd_coeff[['pop2_', 'sex', 'parm', 'estimate']]
ckd_coeff['sex'] = np.where(ckd_coeff['sex'] ==1, 'male', 'female')
ckd_coeff['pop2_'] = ckd_coeff['pop2_'].str.lower()
ckd_coeff['parm'] = ckd_coeff['parm'].str.lower()

# Separate collapsed groups
het_hisp = ckd_coeff.loc[ckd_coeff['pop2_'] == 'het hisp + white'].copy()
het_hisp['pop2_'] = 'het_hisp'
het_white = ckd_coeff.loc[ckd_coeff['pop2_'] == 'het hisp + white'].copy()
het_white['pop2_'] = 'het_white'

idu_hisp_male = ckd_coeff.loc[ckd_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_hisp_male['pop2_'] = 'idu_hisp'
idu_white_male = ckd_coeff.loc[ckd_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_white_male['pop2_'] = 'idu_white'

idu_hisp_female = ckd_coeff.loc[ckd_coeff['pop2_'] == 'idu females'].copy()
idu_hisp_female['pop2_'] = 'idu_hisp'
idu_black_female = ckd_coeff.loc[ckd_coeff['pop2_'] == 'idu females'].copy()
idu_black_female['pop2_'] = 'idu_black'
idu_white_female = ckd_coeff.loc[ckd_coeff['pop2_'] == 'idu females'].copy()
idu_white_female['pop2_'] = 'idu_white'

ckd_coeff = ckd_coeff.loc[~ckd_coeff['pop2_'].isin(['het hisp + white', 'idu hisp + white', 'idu females'])].copy()

# Reattach separated groups
ckd_coeff = ckd_coeff.append(het_hisp, ignore_index=True)
ckd_coeff = ckd_coeff.append(het_white, ignore_index=True)
ckd_coeff = ckd_coeff.append(idu_hisp_male, ignore_index=True)
ckd_coeff = ckd_coeff.append(idu_white_male, ignore_index=True)
ckd_coeff = ckd_coeff.append(idu_hisp_female, ignore_index=True)
ckd_coeff = ckd_coeff.append(idu_black_female, ignore_index=True)
ckd_coeff = ckd_coeff.append(idu_white_female, ignore_index=True)

# Clean up dataframe
ckd_coeff['group'] = ckd_coeff['pop2_'] + '_' + ckd_coeff['sex']
ckd_coeff['param'] = ckd_coeff['parm']
ckd_coeff = ckd_coeff.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
print(ckd_coeff)

# Load dm coeff csv file
dm_coeff = pd.read_csv(f'{in_dir}/dm_coeff.csv')

dm_coeff.columns = dm_coeff.columns.str.lower()
dm_coeff = dm_coeff[['pop2_', 'sex', 'parm', 'estimate']]
dm_coeff['sex'] = np.where(dm_coeff['sex'] ==1, 'male', 'female')
dm_coeff['pop2_'] = dm_coeff['pop2_'].str.lower()
dm_coeff['parm'] = dm_coeff['parm'].str.lower()

# Separate collapsed groups
het_hisp = dm_coeff.loc[dm_coeff['pop2_'] == 'het hisp + white'].copy()
het_hisp['pop2_'] = 'het_hisp'
het_white = dm_coeff.loc[dm_coeff['pop2_'] == 'het hisp + white'].copy()
het_white['pop2_'] = 'het_white'

idu_hisp_male = dm_coeff.loc[dm_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_hisp_male['pop2_'] = 'idu_hisp'
idu_white_male = dm_coeff.loc[dm_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_white_male['pop2_'] = 'idu_white'

idu_hisp_female = dm_coeff.loc[dm_coeff['pop2_'] == 'idu females'].copy()
idu_hisp_female['pop2_'] = 'idu_hisp'
idu_black_female = dm_coeff.loc[dm_coeff['pop2_'] == 'idu females'].copy()
idu_black_female['pop2_'] = 'idu_black'
idu_white_female = dm_coeff.loc[dm_coeff['pop2_'] == 'idu females'].copy()
idu_white_female['pop2_'] = 'idu_white'

dm_coeff = dm_coeff.loc[~dm_coeff['pop2_'].isin(['het hisp + white', 'idu hisp + white', 'idu females'])].copy()

# Reattach separated groups
dm_coeff = dm_coeff.append(het_hisp, ignore_index=True)
dm_coeff = dm_coeff.append(het_white, ignore_index=True)
dm_coeff = dm_coeff.append(idu_hisp_male, ignore_index=True)
dm_coeff = dm_coeff.append(idu_white_male, ignore_index=True)
dm_coeff = dm_coeff.append(idu_hisp_female, ignore_index=True)
dm_coeff = dm_coeff.append(idu_black_female, ignore_index=True)
dm_coeff = dm_coeff.append(idu_white_female, ignore_index=True)

# Clean up dataframe
dm_coeff['group'] = dm_coeff['pop2_'] + '_' + dm_coeff['sex']
dm_coeff['param'] = dm_coeff['parm']
diabetes_coeff = dm_coeff.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]

# Load ht coeff csv file
ht_coeff = pd.read_csv(f'{in_dir}/ht_coeff.csv')

ht_coeff.columns = ht_coeff.columns.str.lower()
ht_coeff = ht_coeff[['pop2_', 'sex', 'parm', 'estimate']]
ht_coeff['sex'] = np.where(ht_coeff['sex'] ==1, 'male', 'female')
ht_coeff['pop2_'] = ht_coeff['pop2_'].str.lower()
ht_coeff['parm'] = ht_coeff['parm'].str.lower()

# Separate collapsed groups
het_hisp = ht_coeff.loc[ht_coeff['pop2_'] == 'het hisp + white'].copy()
het_hisp['pop2_'] = 'het_hisp'
het_white = ht_coeff.loc[ht_coeff['pop2_'] == 'het hisp + white'].copy()
het_white['pop2_'] = 'het_white'

idu_hisp_male = ht_coeff.loc[ht_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_hisp_male['pop2_'] = 'idu_hisp'
idu_white_male = ht_coeff.loc[ht_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_white_male['pop2_'] = 'idu_white'

idu_hisp_female = ht_coeff.loc[ht_coeff['pop2_'] == 'idu females'].copy()
idu_hisp_female['pop2_'] = 'idu_hisp'
idu_black_female = ht_coeff.loc[ht_coeff['pop2_'] == 'idu females'].copy()
idu_black_female['pop2_'] = 'idu_black'
idu_white_female = ht_coeff.loc[ht_coeff['pop2_'] == 'idu females'].copy()
idu_white_female['pop2_'] = 'idu_white'

ht_coeff = ht_coeff.loc[~ht_coeff['pop2_'].isin(['het hisp + white', 'idu hisp + white', 'idu females'])].copy()

# Reattach separated groups
ht_coeff = ht_coeff.append(het_hisp, ignore_index=True)
ht_coeff = ht_coeff.append(het_white, ignore_index=True)
ht_coeff = ht_coeff.append(idu_hisp_male, ignore_index=True)
ht_coeff = ht_coeff.append(idu_white_male, ignore_index=True)
ht_coeff = ht_coeff.append(idu_hisp_female, ignore_index=True)
ht_coeff = ht_coeff.append(idu_black_female, ignore_index=True)
ht_coeff = ht_coeff.append(idu_white_female, ignore_index=True)

# Clean up dataframe
ht_coeff['group'] = ht_coeff['pop2_'] + '_' + ht_coeff['sex']
ht_coeff['param'] = ht_coeff['parm']
hypertension_coeff = ht_coeff.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]

# Load dm coeff csv file
lipid_coeff = pd.read_csv(f'{in_dir}/lipid_coeff.csv')

lipid_coeff.columns = lipid_coeff.columns.str.lower()
lipid_coeff = lipid_coeff[['pop2_', 'sex', 'parm', 'estimate']]
lipid_coeff['sex'] = np.where(lipid_coeff['sex'] ==1, 'male', 'female')
lipid_coeff['pop2_'] = lipid_coeff['pop2_'].str.lower()
lipid_coeff['parm'] = lipid_coeff['parm'].str.lower()

# Separate collapsed groups
het_hisp = lipid_coeff.loc[lipid_coeff['pop2_'] == 'het hisp + white'].copy()
het_hisp['pop2_'] = 'het_hisp'
het_white = lipid_coeff.loc[lipid_coeff['pop2_'] == 'het hisp + white'].copy()
het_white['pop2_'] = 'het_white'

idu_hisp_male = lipid_coeff.loc[lipid_coeff['pop2_'] == 'idu males'].copy()
idu_hisp_male['pop2_'] = 'idu_hisp'
idu_white_male = lipid_coeff.loc[lipid_coeff['pop2_'] == 'idu males'].copy()
idu_white_male['pop2_'] = 'idu_white'
idu_black_male = lipid_coeff.loc[lipid_coeff['pop2_'] == 'idu males'].copy()
idu_black_male['pop2_'] = 'idu_black'

idu_hisp_female = lipid_coeff.loc[lipid_coeff['pop2_'] == 'idu females'].copy()
idu_hisp_female['pop2_'] = 'idu_hisp'
idu_black_female = lipid_coeff.loc[lipid_coeff['pop2_'] == 'idu females'].copy()
idu_black_female['pop2_'] = 'idu_black'
idu_white_female = lipid_coeff.loc[lipid_coeff['pop2_'] == 'idu females'].copy()
idu_white_female['pop2_'] = 'idu_white'

lipid_coeff = lipid_coeff.loc[~lipid_coeff['pop2_'].isin(['het hisp + white', 'idu males', 'idu females'])].copy()

# Reattach separated groups
lipid_coeff = lipid_coeff.append(het_hisp, ignore_index=True)
lipid_coeff = lipid_coeff.append(het_white, ignore_index=True)
lipid_coeff = lipid_coeff.append(idu_hisp_male, ignore_index=True)
lipid_coeff = lipid_coeff.append(idu_white_male, ignore_index=True)
lipid_coeff = lipid_coeff.append(idu_black_male, ignore_index=True)
lipid_coeff = lipid_coeff.append(idu_hisp_female, ignore_index=True)
lipid_coeff = lipid_coeff.append(idu_black_female, ignore_index=True)
lipid_coeff = lipid_coeff.append(idu_white_female, ignore_index=True)

# Clean up dataframe
lipid_coeff['group'] = lipid_coeff['pop2_'] + '_' + lipid_coeff['sex']
lipid_coeff['param'] = lipid_coeff['parm']
lipid_coeff = lipid_coeff.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]

# Load anxiety coeff csv file
stage2_prev_users = pd.read_csv(f'{in_dir}/stage2_prev_users.csv')
# Clean up
stage2_prev_users.columns = stage2_prev_users.columns.str.lower()
stage2_prev_users['sex'] = np.where(stage2_prev_users['sex']==1, 'male', 'female')
stage2_prev_users['sex'] = stage2_prev_users['sex'].str.lower()
stage2_prev_users['pop2'] = stage2_prev_users['pop2'].str.lower()
stage2_prev_users['group'] = stage2_prev_users['pop2'] + '_' + stage2_prev_users['sex']
stage2_prev_users['prev'] /= 100.0
stage2_prev_users = stage2_prev_users.rename(columns={'prev': 'proportion'})

ckd_prev_users = stage2_prev_users.loc[stage2_prev_users['dx'] == 'ckd'].reset_index()[['group', 'proportion']].copy()
lipid_prev_users = stage2_prev_users.loc[stage2_prev_users['dx'] == 'lipid'].reset_index()[['group', 'proportion']].copy()
diabetes_prev_users = stage2_prev_users.loc[stage2_prev_users['dx'] == 'dm'].reset_index()[['group', 'proportion']].copy()
hypertension_prev_users = stage2_prev_users.loc[stage2_prev_users['dx'] == 'ht'].reset_index()[['group', 'proportion']].copy()

stage2_prev_inits = pd.read_csv(f'{in_dir}/ht_dm_ckd_lipid_prev_ini.csv')
# Clean up
stage2_prev_inits.columns = stage2_prev_inits.columns.str.lower()
stage2_prev_inits['sex'] = np.where(stage2_prev_inits['sex']==1, 'male', 'female')
stage2_prev_inits['sex'] = stage2_prev_inits['sex'].str.lower()
stage2_prev_inits['pop2'] = stage2_prev_inits['pop2'].str.lower()
stage2_prev_inits['group'] = stage2_prev_inits['pop2'] + '_' + stage2_prev_inits['sex']
stage2_prev_inits['prevalence'] /= 100.0
stage2_prev_inits = stage2_prev_inits.rename(columns={'prevalence': 'proportion'})

ckd_prev_inits = stage2_prev_inits.loc[stage2_prev_inits['dx'] == 'ckd'].reset_index()[['group', 'proportion']].copy()
lipid_prev_inits = stage2_prev_inits.loc[stage2_prev_inits['dx'] == 'lipid'].reset_index()[['group', 'proportion']].copy()
diabetes_prev_inits = stage2_prev_inits.loc[stage2_prev_inits['dx'] == 'dm'].reset_index()[['group', 'proportion']].copy()
hypertension_prev_inits = stage2_prev_inits.loc[stage2_prev_inits['dx'] == 'ht'].reset_index()[['group', 'proportion']].copy()

# Save them all
ckd_coeff.to_feather(f'{out_dir}/ckd_coeff.feather')
lipid_coeff.to_feather(f'{out_dir}/lipid_coeff.feather')
diabetes_coeff.to_feather(f'{out_dir}/diabetes_coeff.feather')
hypertension_coeff.to_feather(f'{out_dir}/hypertension_coeff.feather')

ckd_prev_users.to_feather(f'{out_dir}/ckd_prev_users.feather')
lipid_prev_users.to_feather(f'{out_dir}/lipid_prev_users.feather')
diabetes_prev_users.to_feather(f'{out_dir}/diabetes_prev_users.feather')
hypertension_prev_users.to_feather(f'{out_dir}/hypertension_prev_users.feather')

ckd_prev_inits.to_feather(f'{out_dir}/ckd_prev_inits.feather')
lipid_prev_inits.to_feather(f'{out_dir}/lipid_prev_inits.feather')
diabetes_prev_inits.to_feather(f'{out_dir}/diabetes_prev_inits.feather')
hypertension_prev_inits.to_feather(f'{out_dir}/hypertension_prev_inits.feather')

ckd_table = ckd_coeff.pivot(index='group', columns='param', values='estimate')[['intercept', 'year', 'age', 'cd4n_entry', 'h1yy_time', 'outcare', 'smoking', 'hcv', 'anx', 'dpr', 'lipid', 'dm', 'ht']]
lipid_table = lipid_coeff.pivot(index='group', columns='param', values='estimate')[['intercept', 'year', 'age', 'cd4n_entry', 'h1yy_time', 'outcare', 'smoking', 'hcv', 'anx', 'dpr', 'ckd', 'dm', 'ht']]
diabetes_table = diabetes_coeff.pivot(index='group', columns='param', values='estimate')[['intercept', 'year', 'age', 'cd4n_entry', 'h1yy_time', 'outcare', 'smoking', 'hcv', 'anx', 'dpr', 'ckd', 'lipid', 'ht']]
hypertension_table = hypertension_coeff.pivot(index='group', columns='param', values='estimate')[['intercept', 'year', 'age', 'cd4n_entry', 'h1yy_time', 'outcare', 'smoking', 'hcv', 'anx', 'dpr', 'ckd', 'lipid', 'dm']]
