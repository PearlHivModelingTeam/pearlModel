# Imports
import os
import pandas as pd
import numpy as np
from pathlib import Path

IDU_MODIFIER = 2.0

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
in_dir = f'{pearl_dir}/param/raw/'
out_dir = f'{pearl_dir}/param/param/'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

df = pd.read_csv(f'{in_dir}/cdc_mortality.csv')
df = df.loc[df['year'] == 2018]
df = df.loc[df['racecat'] != 'All']
df.loc[df['racecat'] == 'Hispanic', 'racecat'] = 'hisp'
df.loc[df['racecat'] == 'non-Hispanic Black', 'racecat'] = 'black'
df.loc[df['racecat'] == 'non-Hispanic White', 'racecat'] = 'white'
df['gender'] = df['gender'].str.lower()

mortality_age_groups = {i: j for i, j in zip(df['five_year_age_groups'].unique(), np.arange(14))}
print(mortality_age_groups)
df['mortality_age_group'] = df['five_year_age_groups'].replace(to_replace=mortality_age_groups)
df = pd.concat([df.assign(risk='msm'), df.assign(risk='het'), df.assign(risk='idu')])
df.loc[df['risk'] == 'idu', 'ir_1000'] = df.loc[df['risk'] == 'idu', 'ir_1000'] * IDU_MODIFIER
df['p'] = df['ir_1000'] / 1000

df['group'] = df['risk'] + '_' + df['racecat'] + '_' + df['gender']
df = df.loc[~((df['risk'] == 'msm') & (df['gender'] == 'female'))]
df = df[['group', 'mortality_age_group', 'p']]
df.to_csv(f'{out_dir}/cdc_mortality.csv', index=False)
print(df.group.unique())

