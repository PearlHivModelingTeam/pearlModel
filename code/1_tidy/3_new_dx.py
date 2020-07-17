# Imports
import os
import pandas as pd

# Define directories
input_dir = f'{os.getcwd()}/../../data/input/aim1'

races = ['black', 'white', 'hisp']

df = pd.read_csv(f'{input_dir}/dx_estimates_cdc_table1.csv')
df.columns = df.columns.str.lower()
df.loc[df['sex']=='Males', 'sex'] = 'male'
df.loc[df['sex']=='Females', 'sex'] = 'female'
df.loc[df['race']=='White Non-Hispanic', 'race'] = 'white'
df.loc[df['race']=='Black Non-Hispanic', 'race'] = 'black'
df.loc[df['race']=='Hispanic', 'race'] = 'hisp'
df = df.loc[(df['race'] != 'Other') & (df['risk'] != 'other')].copy()
df['group'] = df['risk'] + '_' + df['race'] + '_' + df['sex']
del df['risk'], df['race'], df['sex']

df = pd.DataFrame(df.groupby(['group', 'year'])['n_hivdx_cdctable1'].sum())
df = df.rename(columns={'n_hivdx_cdctable1': 'n_dx'})

df_2 = pd.read_csv(f'{input_dir}/new_dx_2018.csv').set_index(['group', 'year'])
df = pd.concat([df, df_2]).sort_index().reset_index()

df.to_csv(f'{input_dir}/new_dx.csv', index=False)




