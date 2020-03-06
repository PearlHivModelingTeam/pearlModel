# Imports
import os
import numpy as np
import pandas as pd
import feather
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", 1001)

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../../data/input/aim_2/stage_1'
out_dir = cwd + '/../../data/parameters/aim_2/stage_1'

# Load depression coeff csv file
depression_coeff = pd.read_csv(f'{in_dir}/depression_coeff.csv')

# Clean dataframe
depression_coeff.columns = depression_coeff.columns.str.lower()
depression_coeff = depression_coeff[['pop2_', 'sex', 'parm', 'estimate']]
depression_coeff['sex'] = np.where(depression_coeff['sex'] ==1, 'male', 'female')
depression_coeff['pop2_'] = depression_coeff['pop2_'].str.lower()
depression_coeff['parm'] = depression_coeff['parm'].str.lower()

# Separate collapsed groups
idu_hisp_male = depression_coeff.loc[depression_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_white_male = depression_coeff.loc[depression_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_black_female = depression_coeff.loc[depression_coeff['pop2_'] == 'idu females'].copy()
idu_hisp_female = depression_coeff.loc[depression_coeff['pop2_'] == 'idu females'].copy()
idu_white_female = depression_coeff.loc[depression_coeff['pop2_'] == 'idu females'].copy()
idu_hisp_male['pop2_'] = 'idu_hisp'
idu_white_male['pop2_'] = 'idu_white'
idu_hisp_female['pop2_'] = 'idu_hisp'
idu_white_female['pop2_'] = 'idu_white'
idu_black_female['pop2_'] = 'idu_black'
depression_coeff = depression_coeff.loc[~depression_coeff['pop2_'].isin(['idu hisp + white', 'idu females'])].copy()

# Reattach separated groups
depression_coeff = depression_coeff.append(idu_hisp_female, ignore_index=True)
depression_coeff = depression_coeff.append(idu_white_female, ignore_index=True)
depression_coeff = depression_coeff.append(idu_black_female, ignore_index=True)
depression_coeff = depression_coeff.append(idu_hisp_male, ignore_index=True)
depression_coeff = depression_coeff.append(idu_white_male, ignore_index=True)

# Clean up dataframe
depression_coeff['group'] = depression_coeff['pop2_'] + '_' + depression_coeff['sex']
depression_coeff['param'] = depression_coeff['parm']
depression_coeff = depression_coeff.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]

# Load anxiety coeff csv file
anxiety_coeff = pd.read_csv(f'{in_dir}/anxiety_coeff.csv')

# Clean dataframe
anxiety_coeff.columns = anxiety_coeff.columns.str.lower()
anxiety_coeff = anxiety_coeff[['pop2_', 'sex', 'parm', 'estimate']]
anxiety_coeff['sex'] = np.where(anxiety_coeff['sex'] ==1, 'male', 'female')
anxiety_coeff['pop2_'] = anxiety_coeff['pop2_'].str.lower()
anxiety_coeff['parm'] = anxiety_coeff['parm'].str.lower()

# Separate collapsed groups
idu_hisp_female = anxiety_coeff.loc[anxiety_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_white_female = anxiety_coeff.loc[anxiety_coeff['pop2_'] == 'idu hisp + white'].copy()
idu_hisp_female['pop2_'] = 'idu_hisp'
idu_white_female['pop2_'] = 'idu_white'
anxiety_coeff = anxiety_coeff.loc[~anxiety_coeff['pop2_'].isin(['idu hisp + white'])].copy()

# Reattach separated groups
anxiety_coeff = anxiety_coeff.append(idu_hisp_female, ignore_index=True)
anxiety_coeff = anxiety_coeff.append(idu_white_female, ignore_index=True)

# Clean up dataframe
anxiety_coeff['group'] = anxiety_coeff['pop2_'] + '_' + anxiety_coeff['sex']
anxiety_coeff['param'] = anxiety_coeff['parm']
anxiety_coeff = anxiety_coeff.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
#anxiety_coeff.to_csv(f'{param_dir}/stage_1/anxiety_coeff.csv', index=False)

# Load anxiety coeff csv file
mh_prev_users = pd.read_csv(f'{in_dir}/mh_prev_users.csv')

# Clean up
mh_prev_users.columns = mh_prev_users.columns.str.lower()
mh_prev_users['sex'] = np.where(mh_prev_users['sex']==1, 'male', 'female')
mh_prev_users['sex'] = mh_prev_users['sex'].str.lower()
mh_prev_users['pop2'] = mh_prev_users['pop2'].str.lower()
mh_prev_users['group'] = mh_prev_users['pop2'] + '_' + mh_prev_users['sex']
mh_prev_users['prev'] /= 100.0
mh_prev_users = mh_prev_users.rename(columns={'prev': 'proportion'})

# Separate mental health conditions
anxiety_prev_users = mh_prev_users.loc[mh_prev_users['dx'] == 'anx'].reset_index()[['group', 'proportion']].copy()
depression_prev_users = mh_prev_users.loc[mh_prev_users['dx'] == 'dpr'].reset_index()[['group', 'proportion']].copy()

mh_prev_inits = pd.read_csv(f'{in_dir}/mh_prev_ini.csv')
mh_prev_inits['sex'] = np.where(mh_prev_inits['sex']==1, 'male', 'female')
mh_prev_inits['sex'] = mh_prev_inits['sex'].str.lower()
mh_prev_inits['pop2'] = mh_prev_inits['pop2'].str.lower()
mh_prev_inits['group'] = mh_prev_inits['pop2'] + '_' + mh_prev_inits['sex']
mh_prev_inits['prevalence'] /= 100.0
mh_prev_inits = mh_prev_inits.rename(columns={'prevalence': 'proportion'}).sort_values(['dx','group'])

anxiety_prev_inits = mh_prev_inits.loc[mh_prev_inits['dx'] == 'anx'].reset_index()[['group', 'proportion']].copy()
depression_prev_inits = mh_prev_inits.loc[mh_prev_inits['dx'] == 'dpr'].reset_index()[['group', 'proportion']].copy()

# Save them all
anxiety_coeff.to_feather(f'{out_dir}/anxiety_coeff.feather')
depression_coeff.to_feather(f'{out_dir}/depression_coeff.feather')
anxiety_prev_users.to_feather(f'{out_dir}/anxiety_prev_users.feather')
depression_prev_users.to_feather(f'{out_dir}/depression_prev_users.feather')
anxiety_prev_inits.to_feather(f'{out_dir}/anxiety_prev_inits.feather')
depression_prev_inits.to_feather(f'{out_dir}/depression_prev_inits.feather')
