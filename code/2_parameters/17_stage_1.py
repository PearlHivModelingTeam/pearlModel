# Imports
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
in_dir = f'{pearl_dir}/param/raw/aim2/stage1'
out_dir = f'{pearl_dir}/param/param/aim2/stage1'

# Depression coeffs
df = pd.read_csv(f'{in_dir}/depression_coeff.csv')
df.columns = df.columns.str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['sex'] = np.where(df['sex'] == 1, 'male', 'female')
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()
df1 = df.loc[df['pop2_'] == 'idu females'].copy()
df = df.loc[df['pop2_'] != 'idu females'].copy()
df = pd.concat([df, df1.assign(pop2_='idu_white'), df1.assign(pop2_='idu_black'), df1.assign(pop2_='idu_hisp')])
df['group'] = df['pop2_'] + '_' + df['sex']
df['param'] = df['parm']
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/depression_coeff.csv', index=False)

# Anxiety coeffs
df = pd.read_csv(f'{in_dir}/anxiety_coeff.csv')
df.columns = df.columns.str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['sex'] = np.where(df['sex'] ==1, 'male', 'female')
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()

df1 = df.loc[df['pop2_'] == 'het_hisp + het_white'].copy()
df = df.loc[df['pop2_'] != 'het_hisp + het_white'].copy()
df2 = df.loc[df['pop2_'] == 'idu_hisp + idu_white'].copy()
df = df.loc[df['pop2_'] != 'idu_hisp + idu_white'].copy()
df = pd.concat([df, df1.assign(pop2_='het_hisp'), df1.assign(pop2_='het_white'),
                df2.assign(pop2_='idu_hisp'), df2.assign(pop2_='idu_white')])
df['group'] = df['pop2_'] + '_' + df['sex']
df['param'] = df['parm']
df = df.copy().sort_values(by=['group', 'param']).reset_index()[['group', 'param', 'estimate']]
df.to_csv(f'{out_dir}/anxiety_coeff.csv', index=False)

# Anxiety and depression prevalence users
df = pd.read_csv(f'{in_dir}/mh_prev_users.csv')

df.columns = df.columns.str.lower()
df['sex'] = np.where(df['sex']==1, 'male', 'female')
df['sex'] = df['sex'].str.lower()
df['pop2'] = df['pop2'].str.lower()
df['group'] = df['pop2'] + '_' + df['sex']
df['prev'] /= 100.0
df = df.rename(columns={'prev': 'proportion'})
df1 = df.loc[df['dx'] == 'dpr'].reset_index()[['group', 'proportion']].copy()
df2 = df.loc[df['dx'] == 'anx'].reset_index()[['group', 'proportion']].copy()

df1.to_csv(f'{out_dir}/depression_prev_users.csv', index=False)
df2.to_csv(f'{out_dir}/anxiety_prev_users.csv', index=False)

# Anxiety and depression prevalence initiators
df = pd.read_csv(f'{in_dir}/mh_prev_ini.csv')

df['sex'] = np.where(df['sex']==1, 'male', 'female')
df['sex'] = df['sex'].str.lower()
df['pop2'] = df['pop2'].str.lower()
df['group'] = df['pop2'] + '_' + df['sex']
df['prevalence'] /= 100.0
df = df.rename(columns={'prevalence': 'proportion'}).sort_values(['dx','group'])
df1 = df.loc[df['dx'] == 'dpr'].reset_index()[['group', 'proportion']].copy()
df2 = df.loc[df['dx'] == 'anx'].reset_index()[['group', 'proportion']].copy()

df1.to_csv(f'{out_dir}/depression_prev_inits.csv', index=False)
df2.to_csv(f'{out_dir}/anxiety_prev_inits.csv', index=False)
