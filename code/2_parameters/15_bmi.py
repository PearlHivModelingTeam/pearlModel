# Imports
import os
import pandas as pd
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
in_dir = f'{pearl_dir}/param/raw/aim2/bmi'
out_dir = f'{pearl_dir}/param/param/aim2/bmi'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

# Pre art coeffs
df = pd.read_csv(f'{in_dir}/pre_art_bmi_coeff_final.csv')
df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop'] = df['pop'].str.lower()
df['parameter'] = df['parameter'].str.lower()
df1 = df.loc[df['pop'] == 'idu_hisp_white'].copy()
df = df.loc[df['pop'] != 'idu_hisp_white'].copy()
df1 = pd.concat([df1.assign(pop='idu_hisp'), df1.assign(pop='idu_white')])
df = pd.concat([df, df1], ignore_index=True)
df['group'] = df['pop'] + '_' + df['sex']
del df['pop'], df['sex']
df = df[['group', 'model', 'parameter', 'coeff']]
df = df.set_index(['group', 'parameter']).sort_index().reset_index()
df1 = df[['group', 'model']]
df1 = df1.drop_duplicates().set_index('group').reset_index()
df1.to_csv(f'{out_dir}/pre_art_bmi_model.csv', index=False)
del df['model']
df.to_csv(f'{out_dir}/pre_art_bmi.csv', index=False)

# Pre art age knots
df = pd.read_csv(f'{in_dir}/pre_art_bmi_age_knots_final.csv')
df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop'] = df['pop'].str.lower()
df1 = df.loc[df['pop'] == 'idu_hisp_white'].copy()
df = df.loc[df['pop'] != 'idu_hisp_white'].copy()
df1 = pd.concat([df1.assign(pop='idu_hisp'), df1.assign(pop='idu_white')])
df = pd.concat([df, df1], ignore_index=True)
df['group'] = df['pop'] + '_' + df['sex']
del df['pop'], df['sex']
df = df[['group', 'knot_number', 'knot']]
df = df.pivot_table(index='group', columns='knot_number', values='knot').reset_index()
df.columns = df.columns.astype(str)
df.to_csv(f'{out_dir}/pre_art_bmi_age_knots.csv', index=False)

# Pre art h1yy knots
df = pd.read_csv(f'{in_dir}/pre_art_bmi_h1yy_knots_final.csv')
df.columns = df.columns.str.lower()
df['sex'] = df['sex'].str.lower()
df['pop'] = df['pop'].str.lower()
df1 = df.loc[df['pop'] == 'idu_hisp_white'].copy()
df = df.loc[df['pop'] != 'idu_hisp_white'].copy()
df1 = pd.concat([df1.assign(pop='idu_hisp'), df1.assign(pop='idu_white')])
df = pd.concat([df, df1], ignore_index=True)
df['group'] = df['pop'] + '_' + df['sex']
del df['pop'], df['sex']
df = df[['group', 'knot_number', 'knot']]
df = df.pivot_table(index='group', columns='knot_number', values='knot').reset_index()
df.columns = df.columns.astype(str)
df.to_csv(f'{out_dir}/pre_art_bmi_h1yy_knots.csv', index=False)

# Post art coeffs
df = pd.read_csv(f'{in_dir}/post_art_bmi_coeffs.csv')
df.columns = df.columns.str.lower()
df = df.rename(columns={'pop2': 'pop', 'variable': 'parameter', 'beta': 'coeff'})
df['sex'] = df['sex'].str.lower()
df['pop'] = df['pop'].str.lower()
df['parameter'] = df['parameter'].str.lower()
df1 = df.loc[df['pop'] == 'idu_hisp_white'].copy()
df = df.loc[df['pop'] != 'idu_hisp_white'].copy()
df1 = pd.concat([df1.assign(pop='idu_hisp'), df1.assign(pop='idu_white')])
df = pd.concat([df, df1], ignore_index=True)
df['group'] = df['pop'] + '_' + df['sex']
del df['pop'], df['sex']
df = df[['group', 'parameter', 'coeff']]
df = df.set_index(['group', 'parameter']).sort_index().reset_index()
df.to_csv(f'{out_dir}/post_art_bmi.csv', index=False)

# Post art knots
df = pd.read_csv(f'{in_dir}/post_art_bmi_knot_locations.csv')
df = df.rename(columns={'pop2': 'pop', 'var': 'parameter'})
df['sex'] = df['sex'].str.lower()
df['pop'] = df['pop'].str.lower()
df1 = df.loc[df['pop'] == 'idu_hisp_white'].copy()
df = df.loc[df['pop'] != 'idu_hisp_white'].copy()
df1 = pd.concat([df1.assign(pop='idu_hisp'), df1.assign(pop='idu_white')])
df = pd.concat([df, df1], ignore_index=True)
df['group'] = df['pop'] + '_' + df['sex']
del df['pop'], df['sex']
df.loc[df['parameter'] == 'sqrt(pre-ART BMI)', 'parameter'] = 'pre_sqrt'
df.loc[df['parameter'] == 'sqrt(CD4) at ART initiation', 'parameter'] = 'sqrtcd4'
df.loc[df['parameter'] == 'sqrt(CD4) 1-2 years after ART initiation', 'parameter'] = 'sqrtcd4_post'
df = df.set_index(['parameter', 'group', 'knotnum']).sort_index()

# age
df1 = df.loc['age'].reset_index().pivot_table(index='group', columns='knotnum', values='knotvalue').reset_index()
df1.columns = df1.columns.astype(str)
df1.to_csv(f'{out_dir}/post_art_bmi_age_knots.csv', index=False)

# pre_sqrt
df1 = df.loc['pre_sqrt'].reset_index().pivot_table(index='group', columns='knotnum', values='knotvalue').reset_index()
df1.columns = df1.columns.astype(str)
df1.to_csv(f'{out_dir}/post_art_bmi_pre_art_bmi_knots.csv', index=False)

# sqrtcd4
df1 = df.loc['sqrtcd4'].reset_index().pivot_table(index='group', columns='knotnum', values='knotvalue').reset_index()
df1.columns = df1.columns.astype(str)
df1.to_csv(f'{out_dir}/post_art_bmi_cd4_knots.csv', index=False)

# sqrtcd4_post
df1 = df.loc['sqrtcd4_post'].reset_index().pivot_table(index='group', columns='knotnum', values='knotvalue').reset_index()
df1.columns = df1.columns.astype(str)
df1.to_csv(f'{out_dir}/post_art_bmi_cd4_post_knots.csv', index=False)
