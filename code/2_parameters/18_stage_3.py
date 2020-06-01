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
in_dir = cwd + '/../../data/input/aim2/stage3'
out_dir = cwd + '/../../data/parameters/aim2/stage3'

# Malignancy incidence coeff
df = pd.read_csv(f'{in_dir}/malig/malig_coeff.csv')
df.columns = df.columns.str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['sex'] = np.where(df['sex'] ==1, 'male', 'female')
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()
df['group'] = df['pop2_'] + '_' + df['sex']
df = df.rename(columns = {'parm': 'param'})
del df['pop2_'], df['sex']
df = df.pivot_table(index = 'group', columns='param', values='estimate').reset_index()
df = pd.concat([df] + [df.loc[df['group'] == 'all women_female'].assign(group = group_name) for group_name in group_names if ('female' in group_name)], ignore_index=True)
df = pd.concat([df] + [df.loc[df['group'] == 'het hisp + white_male'].assign(group = group_name) for group_name in ['het_hisp_male', 'het_white_male']], ignore_index=True)
df = pd.concat([df] + [df.loc[df['group'] == 'idu hisp + white_male'].assign(group = group_name) for group_name in ['idu_hisp_male', 'idu_white_male']], ignore_index=True)
df = df.loc[~df['group'].isin(['all women_female', 'het hisp + white_male', 'idu hisp + white_male'])].set_index('group').sort_index().reset_index()
df.to_feather(f'{out_dir}/malig_coeff.feather')

# Esld incidence coeff
df = pd.read_csv(f'{in_dir}/esld/esld_coeff.csv')
df.columns = df.columns.str.lower()
df = df.rename(columns = {'parm': 'param'})
df['param'] = df['param'].str.lower()
df = df[['param', 'estimate']]
df['group'] = 'all'
df = df.pivot_table(index='group', columns='param', values='estimate').reset_index()
df = pd.concat([df.assign(group = group_name) for group_name in group_names], ignore_index=True)
df.to_feather(f'{out_dir}/esld_coeff.feather')

# MI incidence coeff
df = pd.read_csv(f'{in_dir}/mi/mi_coeff.csv')
df.columns = df.columns.str.lower()
df = df[['pop2_', 'parm', 'estimate']]
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()
df['group'] = df['pop2_']
df.loc[df['group']=='msm_white', 'group'] = 'msm_white_male'
df = df.rename(columns = {'parm': 'param'})
del df['pop2_']
df = df.pivot_table(index = 'group', columns='param', values='estimate').reset_index()
df = pd.concat([df] + [df.loc[df['group'] == 'everyone else'].assign(group = group_name) for group_name in group_names if group_name != 'msm_white_male'], ignore_index=True)
df = df.loc[df['group'] != 'everyone else'].copy().set_index('group').sort_index().reset_index()
df.to_feather(f'{out_dir}/mi_coeff.feather')

# Malignancy prevalence users
df = pd.read_csv(f'{in_dir}/malig/malig_prev_users_2009.csv')
df.columns = df.columns.str.lower()
df['sex'] = np.where(df['sex'] ==1, 'male', 'female')
df['pop2'] = df['pop2'].str.lower()
df['group'] = df['pop2'] + '_' + df['sex']
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']]
df.to_feather(f'{out_dir}/malig_prev_users.feather')

# esld prevalence users
df = pd.read_csv(f'{in_dir}/esld/esld_prev_users_2009.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop2'].str.lower()
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']]
df = pd.concat([df.loc[df['group'] == 'all msm'].assign(group = group_name) for group_name in group_names if 'msm' in group_name] +
               [df.loc[df['group'] == 'everyone else'].assign(group = group_name) for group_name in group_names if 'msm' not in group_name], ignore_index=True).set_index('group').sort_index().reset_index()
df.to_feather(f'{out_dir}/esld_prev_users.feather')

# mi prevalence users
df = pd.read_csv(f'{in_dir}/mi/mi_prev_users_2009.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop2'].str.lower()
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']]
df = pd.concat([df.loc[df['group'] == 'msm_white'].assign(group = group_name) for group_name in group_names if 'msm_white' in group_name] +
               [df.loc[df['group'] == 'everyone else'].assign(group = group_name) for group_name in group_names if 'msm_white' not in group_name], ignore_index=True).set_index('group').sort_index().reset_index()
df.to_feather(f'{out_dir}/mi_prev_users.feather')

# malignancy prevalence initiators
df = pd.read_csv(f'{in_dir}/malig/malig_prev_ini.csv')
df.columns = df.columns.str.lower()
df['sex'] = np.where(df['sex'] ==1, 'male', 'female')
df['pop2'] = df['pop2'].str.lower()
df['group'] = df['pop2'] + '_' + df['sex']
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']].set_index('group').sort_index().reset_index()
df.to_feather(f'{out_dir}/malig_prev_inits.feather')

# esld prevalence initiators
df = pd.read_csv(f'{in_dir}/esld/esld_prev_ini.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop2'].str.lower()
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']]
df = pd.concat([df.assign(group = group_name) for group_name in group_names]).set_index('group').sort_index().reset_index()
df.to_feather(f'{out_dir}/esld_prev_inits.feather')

# mi prevalence initiators
df = pd.read_csv(f'{in_dir}/mi/MI_prev_ini.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop2'].str.lower()
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']]
df = pd.concat([df.assign(group = group_name) for group_name in group_names]).set_index('group').sort_index().reset_index()
df.to_feather(f'{out_dir}/mi_prev_inits.feather')

