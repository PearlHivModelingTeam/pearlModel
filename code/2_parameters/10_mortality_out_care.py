# Imports
import os
import pandas as pd
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
in_dir = f'{pearl_dir}/param/raw/'
out_dir = f'{pearl_dir}/param/param/'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

df = pd.read_csv(f'{in_dir}/mortality_out_care_coeffs.csv')
df.columns = df.columns.str.lower()
df['pop3'] = df['pop3'].str.lower()
df['parm'] = df['parm'].str.lower()
df = df[['pop3', 'parm', 'estimate']]
print(df.pop3.unique())
df1 = df.loc[df['pop3'] == 'idu_hisp + idu_white_women'].copy()
df1 = pd.concat([df1.assign(pop3='idu_hisp_women'), df1.assign(pop3='idu_white_women')])
df2 = df.loc[df['pop3'] == 'het_hisp + het_white_women'].copy()
df2 = pd.concat([df2.assign(pop3='het_hisp_women'), df2.assign(pop3='het_white_women')])
df3 = df.loc[df['pop3'] == 'het_hisp + het_white_men'].copy()
df3 = pd.concat([df3.assign(pop3='het_hisp_men'), df3.assign(pop3='het_white_men')])
df = df.loc[(df['pop3'] != 'idu_hisp + idu_white_women') & (df['pop3'] != 'het_hisp + het_white_women') & (df['pop3'] != 'het_hisp + het_white_men')].copy()
df = pd.concat([df, df1, df2, df3], ignore_index=True)
df['sex'] = '-'
df.loc[df['pop3'].str.contains('_men'), 'sex'] = 'male'
df.loc[df['pop3'].str.contains('_women'), 'sex'] = 'female'
group_split = df['pop3'].str.split('_', expand=True)
df['group'] = group_split[0] + '_' + group_split[1] + '_' + df['sex']
df = df[['group', 'parm', 'estimate']]
df = df.set_index(['group', 'parm']).sort_index().reset_index()
df = df.pivot_table(index='group', columns='parm', values='estimate').reset_index()
df.to_csv(f'{out_dir}/mortality_out_care.csv', index=False)

df = pd.read_csv(f'{in_dir}/mortality_out_care_knots.csv')
df.columns = df.columns.str.lower()
df['pop2'] = df['pop2'].str.lower()
df['variable'] = df['variable'].str.lower()
df1 = df.loc[df['pop2'] == 'idu_hisp + idu_white'].copy()
df1 = pd.concat([df1.assign(pop2='idu_hisp'), df1.assign(pop2='idu_white')])
df2 = df.loc[df['pop2'] == 'het_hisp + het_white'].copy()
df2 = pd.concat([df2.assign(pop2='het_hisp'), df2.assign(pop2='het_white')])
df = df.loc[(df['pop2'] != 'idu_hisp + idu_white') & (df['pop2'] != 'het_hisp + het_white')].copy()
df = pd.concat([df, df1, df2], ignore_index=True)
df.loc[df['sex'] == 1, 'sex'] = 'male'
df.loc[df['sex'] == 2, 'sex'] = 'female'
df['group'] = df['pop2'] + '_' + df['sex']
df = df[['variable', 'group', 'p5', 'p35', 'p65', 'p95']].set_index(['variable', 'group']).sort_index()
print(df)
df1 = df.loc['age'].rename(columns={'p5': 1, 'p35': 2, 'p65': 3, 'p95': 4}).reset_index()
df1.to_csv(f'{out_dir}/mortality_out_care_age.csv', index=False)
df2 = df.loc['time-varying sqrt(cd4)'].rename(columns={'p5': 1, 'p35': 2, 'p65': 3, 'p95': 4}).reset_index()
df2.to_csv(f'{out_dir}/mortality_out_care_tv_sqrtcd4.csv', index=False)
