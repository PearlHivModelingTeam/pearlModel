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
in_dir = cwd + '/../../data/input/aim2/mortality'
out_dir = cwd + '/../../data/parameters/aim2/mortality'

# mortality in care coeffs
df = pd.read_csv(f'{in_dir}/mortality_in_care_coeff.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop3'].str.lower()
df['param'] = df['parm'].str.lower()
df = df[['group', 'param', 'estimate']]

group = df['group'].values[0]
df1 = df.loc[df['group'] == group].copy()
df1 = pd.concat([df1.assign(group='het_hisp_female'), df1.assign(group='het_white_female'), df1.assign(group='het_black_female'),
                 df1.assign(group='idu_hisp_female'), df1.assign(group='idu_white_female'), df1.assign(group='idu_black_female')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df2 = df.loc[df['group'] == group].copy()
df2 = pd.concat([df2.assign(group='het_hisp_male'), df2.assign(group='het_white_male'), df2.assign(group='het_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df3 = df.loc[df['group'] == group].copy()
df3 = pd.concat([df3.assign(group='idu_hisp_male'), df3.assign(group='idu_white_male'), df3.assign(group='idu_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].unique()[1]
df4 = df.loc[df['group'] == group].copy()
df4 = pd.concat([df4.assign(group='msm_hisp_male'), df4.assign(group='msm_white_male')])
df = df.loc[df['group'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/mortality_in_care.csv', index=False)

# mortality in care knots
df = pd.read_csv(f'{in_dir}/mortality_in_care_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop3'].str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'

group = df['group'].values[0]
df1 = df.loc[df['group'] == group].copy()
df1 = pd.concat([df1.assign(group='het_hisp_female'), df1.assign(group='het_white_female'), df1.assign(group='het_black_female'),
                 df1.assign(group='idu_hisp_female'), df1.assign(group='idu_white_female'), df1.assign(group='idu_black_female')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df2 = df.loc[df['group'] == group].copy()
df2 = pd.concat([df2.assign(group='het_hisp_male'), df2.assign(group='het_white_male'), df2.assign(group='het_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df3 = df.loc[df['group'] == group].copy()
df3 = pd.concat([df3.assign(group='idu_hisp_male'), df3.assign(group='idu_white_male'), df3.assign(group='idu_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].unique()[1]
df4 = df.loc[df['group'] == group].copy()
df4 = pd.concat([df4.assign(group='msm_hisp_male'), df4.assign(group='msm_white_male')])
df = df.loc[df['group'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/mortality_in_care_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/mortality_in_care_post_art_bmi.csv', index=False)

# mortality out care coeffs
df = pd.read_csv(f'{in_dir}/mortality_out_care_coeff.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop3'].str.lower()
df['param'] = df['parm'].str.lower()
df = df[['group', 'param', 'estimate']]

group = df['group'].values[0]
df1 = df.loc[df['group'] == group].copy()
df1 = pd.concat([df1.assign(group='het_hisp_female'), df1.assign(group='het_white_female'), df1.assign(group='het_black_female'),
                 df1.assign(group='idu_hisp_female'), df1.assign(group='idu_white_female'), df1.assign(group='idu_black_female')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df2 = df.loc[df['group'] == group].copy()
df2 = pd.concat([df2.assign(group='het_hisp_male'), df2.assign(group='het_white_male'), df2.assign(group='het_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df3 = df.loc[df['group'] == group].copy()
df3 = pd.concat([df3.assign(group='idu_hisp_male'), df3.assign(group='idu_white_male'), df3.assign(group='idu_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].unique()[1]
df4 = df.loc[df['group'] == group].copy()
df4 = pd.concat([df4.assign(group='msm_hisp_male'), df4.assign(group='msm_white_male')])
df = df.loc[df['group'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/mortality_out_care.csv', index=False)

# mortality out care knots
df = pd.read_csv(f'{in_dir}/mortality_out_care_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['group'] = df['pop3'].str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'

group = df['group'].values[0]
df1 = df.loc[df['group'] == group].copy()
df1 = pd.concat([df1.assign(group='het_hisp_female'), df1.assign(group='het_white_female'), df1.assign(group='het_black_female'),
                 df1.assign(group='idu_hisp_female'), df1.assign(group='idu_white_female'), df1.assign(group='idu_black_female')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df2 = df.loc[df['group'] == group].copy()
df2 = pd.concat([df2.assign(group='het_hisp_male'), df2.assign(group='het_white_male'), df2.assign(group='het_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].values[0]
df3 = df.loc[df['group'] == group].copy()
df3 = pd.concat([df3.assign(group='idu_hisp_male'), df3.assign(group='idu_white_male'), df3.assign(group='idu_black_male')])
df = df.loc[df['group'] != group].copy()

group = df['group'].unique()[1]
df4 = df.loc[df['group'] == group].copy()
df4 = pd.concat([df4.assign(group='msm_hisp_male'), df4.assign(group='msm_white_male')])
df = df.loc[df['group'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/mortality_out_care_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/mortality_out_care_post_art_bmi.csv', index=False)
