# Imports
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
in_dir = f'{pearl_dir}/param/raw/aim2/stage2'
out_dir = f'{pearl_dir}/param/param/aim2/stage2'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

# ckd coeff
df = pd.read_csv(f'{in_dir}/ckd_coeff.csv')

df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()

group = 'het_hisp + het_white + het_black'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white + idu_black'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white'), df3.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df4 = df.loc[df['pop2_'] == group].copy()
df4 = pd.concat([df4.assign(pop2_='idu_hisp'), df4.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'msm_hisp + msm_white'
df5 = df.loc[df['pop2_'] == group].copy()
df5 = pd.concat([df5.assign(pop2_='msm_hisp'), df5.assign(pop2_='msm_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4, df5])
df['group'] = df['pop2_'] + '_' + df['sex']
df['param'] = df['parm']
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/ckd_coeff.csv', index=False)

# ckd knots
df = pd.read_csv(f'{in_dir}/ckd_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop2_'] = df['pop2_'].str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'

group = 'het_hisp + het_white + het_black'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white + idu_black'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white'), df3.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df4 = df.loc[df['pop2_'] == group].copy()
df4 = pd.concat([df4.assign(pop2_='idu_hisp'), df4.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'msm_hisp + msm_white'
df5 = df.loc[df['pop2_'] == group].copy()
df5 = pd.concat([df5.assign(pop2_='msm_hisp'), df5.assign(pop2_='msm_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4, df5], ignore_index=True)
df['group'] = df['pop2_'] + '_' + df['sex']
df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/ckd_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/ckd_post_art_bmi.csv', index=False)

# dm coeff
df = pd.read_csv(f'{in_dir}/dm_coeff.csv')

df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white + het_black'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white'), df2.assign(pop2_='het_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'msm_hisp + msm_white'
df4 = df.loc[df['pop2_'] == group].copy()
df4 = pd.concat([df4.assign(pop2_='msm_hisp'), df4.assign(pop2_='msm_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df['group'] = df['pop2_'] + '_' + df['sex']
df['param'] = df['parm']
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/dm_coeff.csv', index=False)

# dm knots
df = pd.read_csv(f'{in_dir}/dm_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop2_'] = df['pop2_'].str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white + het_black'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white'), df2.assign(pop2_='het_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'msm_hisp + msm_white'
df4 = df.loc[df['pop2_'] == group].copy()
df4 = pd.concat([df4.assign(pop2_='msm_hisp'), df4.assign(pop2_='msm_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df['group'] = df['pop2_'] + '_' + df['sex']

df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/dm_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/dm_post_art_bmi.csv', index=False)

# ht coeff
df = pd.read_csv(f'{in_dir}/ht_coeff.csv')

df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'msm_hisp + msm_white'
df4 = df.loc[df['pop2_'] == group].copy()
df4 = pd.concat([df4.assign(pop2_='msm_hisp'), df4.assign(pop2_='msm_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df['group'] = df['pop2_'] + '_' + df['sex']
df['param'] = df['parm']
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/ht_coeff.csv', index=False)

# ht knots
df = pd.read_csv(f'{in_dir}/ht_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop2_'] = df['pop2_'].str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

group = 'msm_hisp + msm_white'
df4 = df.loc[df['pop2_'] == group].copy()
df4 = pd.concat([df4.assign(pop2_='msm_hisp'), df4.assign(pop2_='msm_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3, df4])
df['group'] = df['pop2_'] + '_' + df['sex']

df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/ht_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/ht_post_art_bmi.csv', index=False)

# lipid coeff
df = pd.read_csv(f'{in_dir}/lipid_coeff.csv')

df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white + het_black'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white'), df2.assign(pop2_='het_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3])
df['group'] = df['pop2_'] + '_' + df['sex']
df['param'] = df['parm']
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/lipid_coeff.csv', index=False)

# lipid knots
df = pd.read_csv(f'{in_dir}/lipid_bmi_percentiles.csv')
df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop2_'] = df['pop2_'].str.lower()
df['variable'] = df['variable'].str.lower()
df.loc[df['variable'] == 'post-art bmi - pre-art bmi', 'variable'] = 'delta bmi'
df.loc[df['variable'] == 'post-art bmi', 'variable'] = 'post art bmi'

group = 'all women'
df1 = df.loc[df['pop2_'] == group].copy()
df1 = pd.concat([df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'), df1.assign(pop2_='het_black'),
                 df1.assign(pop2_='idu_hisp'), df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'het_hisp + het_white + het_black'
df2 = df.loc[df['pop2_'] == group].copy()
df2 = pd.concat([df2.assign(pop2_='het_hisp'), df2.assign(pop2_='het_white'), df2.assign(pop2_='het_black')])
df = df.loc[df['pop2_'] != group].copy()

group = 'idu_hisp + idu_white'
df3 = df.loc[df['pop2_'] == group].copy()
df3 = pd.concat([df3.assign(pop2_='idu_hisp'), df3.assign(pop2_='idu_white')])
df = df.loc[df['pop2_'] != group].copy()

df = pd.concat([df, df1, df2, df3])
df['group'] = df['pop2_'] + '_' + df['sex']

df = df.sort_values(['variable', 'group'])[['variable', 'group', 'p5', 'p35', 'p65', 'p95']]
df1 = df.loc[df['variable'] == 'delta bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()
df2 = df.loc[df['variable'] == 'post art bmi'][['group', 'p5', 'p35', 'p65', 'p95']].set_index('group').reset_index()

df1.to_csv(f'{out_dir}/lipid_delta_bmi.csv', index=False)
df2.to_csv(f'{out_dir}/lipid_post_art_bmi.csv', index=False)

# Load prev users csv file
df = pd.read_csv(f'{in_dir}/ht_dm_ckd_lipid_prev_users.csv')

df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop2'] = df['pop2'].str.lower()
df['group'] = df['pop2'] + '_' + df['sex']
df['prev'] /= 100.0
df = df.rename(columns={'prev': 'proportion'})

df1 = df.loc[df['dx'] == 'ckd'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/ckd_prev_users.csv', index=False)

df1 = df.loc[df['dx'] == 'dm'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/dm_prev_users.csv', index=False)

df1 = df.loc[df['dx'] == 'ht'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/ht_prev_users.csv', index=False)

df1 = df.loc[df['dx'] == 'lipid'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/lipid_prev_users.csv', index=False)

# Load prev ini csv file
df = pd.read_csv(f'{in_dir}/ht_dm_ckd_lipid_prev_ini.csv')

df.columns = df.columns.str.lower()

df['sex'] = df['sex'].str.lower()
df['pop2'] = df['pop2'].str.lower()
df['group'] = df['pop2'] + '_' + df['sex']
df['prevalence'] /= 100.0
df = df.rename(columns={'prevalence': 'proportion'})

df1 = df.loc[df['dx'] == 'ckd'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/ckd_prev_inits.csv', index=False)

df1 = df.loc[df['dx'] == 'dm'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/dm_prev_inits.csv', index=False)

df1 = df.loc[df['dx'] == 'ht'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/ht_prev_inits.csv', index=False)

df1 = df.loc[df['dx'] == 'lipid'].reset_index()[['group', 'proportion']].copy()
df1.to_csv(f'{out_dir}/lipid_prev_inits.csv', index=False)
