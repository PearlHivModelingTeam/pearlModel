# Imports
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
in_dir = f'{pearl_dir}/param/raw/aim2/stage3'
out_dir = f'{pearl_dir}/param/param/aim2/stage3'


group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

# Malignancy incidence coeff
df = pd.read_csv(f'{in_dir}/malig_coeff.csv')
df.columns = df.columns.str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['pop2_'] = df['pop2_'].str.lower()
df['sex'] = df['sex'].str.lower()

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = df['pop2_'].values[0]
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white'), df2.assign(pop2_='het_black'),
                 df2.assign(pop2_='idu_hisp'), df2.assign(pop2_='idu_white'), df2.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = df['pop2_'].values[0]
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='msm_hisp'), df3.assign(pop2_='msm_white'), df3.assign(pop2_='msm_black')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3])
df['group'] = df['pop2_'] + '_' + df['sex']
df['param'] = df['parm']
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/malig_coeff.csv', index=False)

# malig knots
df = pd.read_csv(f'{in_dir}/malig_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['sex'] = np.where(df['sex'] == 1, 'male', 'female')
df['pop2_'] = df['pop2_'].str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = df['pop2_'].values[0]
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white'), df2.assign(pop2_='het_black'),
                 df2.assign(pop2_='idu_hisp'), df2.assign(pop2_='idu_white'), df2.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = df['pop2_'].values[0]
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='msm_hisp'), df3.assign(pop2_='msm_white'), df3.assign(pop2_='msm_black')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3])
df['group'] = df['pop2_'] + '_' + df['sex']
df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/malig_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/malig_post_art_bmi.csv', index=False)

# Malignancy prevalence users
df = pd.read_csv(f'{in_dir}/malig_prev_users_2009.csv')
df.columns = df.columns.str.lower()
df['pop2'] = df['pop2'].str.lower()
df['sex'] = df['sex'].str.lower()

group = df['pop2'].values[0]
df1 = df.loc[df['pop2'] == group].copy()
df1 = pd.concat([df1.assign(pop2='het_hisp'), df1.assign(pop2='het_white'), df1.assign(pop2='het_black')])
df = df.loc[df['pop2'] != group].copy()

group = df['pop2'].values[0]
df2 = df.loc[df['pop2'] == group].copy()
df2 = pd.concat([df2.assign(pop2='idu_hisp'), df2.assign(pop2='idu_white'), df2.assign(pop2='idu_black')])
df = df.loc[df['pop2'] != group].copy()

group = df['pop2'].values[0]
df3 = df.loc[df['pop2'] == group].copy()
df3 = pd.concat([df3.assign(pop2='het_hisp'), df3.assign(pop2='het_white'), df3.assign(pop2='het_black'),
                 df3.assign(pop2='idu_hisp'), df3.assign(pop2='idu_white'), df3.assign(pop2='idu_black')])
df = df.loc[df['pop2'] != group].copy()

df = pd.concat([df, df1, df2, df3])
df['group'] = df['pop2'] + '_' + df['sex']
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']].set_index('group').sort_index().reset_index()
df.to_csv(f'{out_dir}/malig_prev_users.csv', index=False)

# Malignancy prevalence ini
df = pd.read_csv(f'{in_dir}/malig_prev_ini.csv')
df['pop2_'] = df['pop2_'].str.lower()
df['sex'] = df['sex'].str.lower()

group = df['pop2_'].values[0]
df1 = df.loc[(df['pop2_'] == group) & (df['sex'] == 'male')].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[~((df['pop2_'] == group) & (df['sex'] == 'male'))].copy()

group = df['pop2_'].values[0]
df2 = df.loc[(df['pop2_'] == group) & (df['sex'] == 'female')].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white'), df2.assign(pop2_='het_black'),
                 df2.assign(pop2_='idu_hisp'), df2.assign(pop2_='idu_white'), df2.assign(pop2_='idu_black')])
df = df.loc[~((df['pop2_'] == group) & (df['sex'] == 'female'))].copy()

group = df['pop2_'].values[1]
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='msm_hisp'), df3.assign(pop2_='msm_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3])
df['group'] = df['pop2_'] + '_' + df['sex']
df['proportion'] = df['prev'] / 100.0
df = df[['group', 'proportion']].set_index('group').sort_index().reset_index()
df.to_csv(f'{out_dir}/malig_prev_ini.csv', index=False)

# mi incidence coeff
df = pd.read_csv(f'{in_dir}/mi_coeff.csv')
df.columns = df.columns.str.lower()
df = df[['parm', 'estimate']]
df = pd.concat([df.assign(group=group) for group in group_names])
df['param'] = df['parm']
df = df.set_index(['group', 'param']).sort_index().reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/mi_coeff.csv', index=False)

# mi knots
df = pd.read_csv(f'{in_dir}/mi_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'
df = pd.concat([df.assign(group=group) for group in group_names])

df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/mi_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/mi_post_art_bmi.csv', index=False)

# mi prevalence users
df = pd.read_csv(f'{in_dir}/mi_prev_users_2009.csv')
df.columns = df.columns.str.lower()
df['pop2'] = df['pop2'].str.lower()
df1 = df.loc[df['pop2'] == 'everyone else']
df1 = pd.concat([df1.assign(group=group_name) for group_name in group_names if group_name != 'msm_white_male'])
df = df.loc[df['pop2'] == 'msm_white'].assign(group='msm_white_male')
df = pd.concat([df, df1])
df['proportion'] = df['prev'] / 100.0
df = df.set_index('group').sort_index().reset_index()[['group', 'proportion']]
df.to_csv(f'{out_dir}/mi_prev_users.csv', index=False)

# mi prevalence ini
df = pd.read_csv(f'{in_dir}/mi_prev_ini.csv')
df = pd.concat([df.assign(group=group_name) for group_name in group_names])
df['proportion'] = df['prev'] / 100.0
df = df.set_index('group').sort_index().reset_index()[['group', 'proportion']]
df.to_csv(f'{out_dir}/mi_prev_ini.csv', index=False)

# esld incidence coeff
df = pd.read_csv(f'{in_dir}/esld_coeff.csv')
df.columns = df.columns.str.lower()
df = pd.concat([df.assign(group=group_name) for group_name in group_names])
df = df[['group', 'parm', 'estimate']]
df = df.rename(columns={'parm': 'param'})
df = df.set_index(['group', 'param']).sort_index().reset_index()
df.to_csv(f'{out_dir}/esld_coeff.csv', index=False)

# esld knots
df = pd.read_csv(f'{in_dir}/esld_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'
df = pd.concat([df.assign(group=group) for group in group_names])

df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/esld_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/esld_post_art_bmi.csv', index=False)

# esld prevalence users
df = pd.read_csv(f'{in_dir}/esld_prev_users_2009.csv')
df.columns = df.columns.str.lower()
df['pop2'] = df['pop2'].str.lower()
df1 = df.loc[df['pop2'] == 'everyone else']
df1 = pd.concat([df1.assign(group=group_name) for group_name in group_names if group_name not in ['msm_white_male', 'msm_black_male', 'msm_hisp_male']])
df2 = df.loc[df['pop2'] == 'all msm']
df2 = pd.concat([df2.assign(group=group_name) for group_name in ['msm_white_male', 'msm_black_male', 'msm_hisp_male']])
df = pd.concat([df1, df2])
df['proportion'] = df['prev'] / 100.0
df = df.set_index('group').sort_index().reset_index()[['group', 'proportion']]
df.to_csv(f'{out_dir}/esld_prev_users.csv', index=False)

# esld prevalence ini
df = pd.read_csv(f'{in_dir}/esld_prev_ini.csv')
df = pd.concat([df.assign(group=group_name) for group_name in group_names])
df['proportion'] = df['prev'] / 100.0
df = df.set_index('group').sort_index().reset_index()[['group', 'proportion']]
df.to_csv(f'{out_dir}/esld_prev_ini.csv', index=False)
