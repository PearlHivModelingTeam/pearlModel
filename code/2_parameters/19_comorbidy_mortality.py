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
in_dir = cwd + '/../../data/input/aim2/mortality'
out_dir = cwd + '/../../data/parameters/aim2/mortality'

# Clean in care coefficients
mortality_in_care = pd.read_csv(f'{in_dir}/mortality_coeff_in_care.csv')
mortality_in_care.columns = mortality_in_care.columns.str.lower()
mortality_in_care['sex'] = np.where(mortality_in_care['sex'] ==1, 'male', 'female')
mortality_in_care['pop2'] = mortality_in_care['pop2'].str.lower()
mortality_in_care['group'] = mortality_in_care['pop2'] + '_' + mortality_in_care['sex']
mortality_in_care = mortality_in_care[['group', 'intercept_c', 'year_c', 'agecat_c', 'sqrtcd4n_c', 'h1yy_c', 'smoking_c', 'hcv_c', 'anx_c', 'dpr_c', 'ht_c', 'dm_c', 'ckd_c', 'lipid_c']].copy()
idu_all_female = mortality_in_care['group'] == 'idu_all_female'
idu_all_female_df = mortality_in_care.loc[idu_all_female].copy()
mortality_in_care = mortality_in_care.loc[~idu_all_female]
mortality_in_care = mortality_in_care.append(idu_all_female_df.assign(group='idu_black_female'))
mortality_in_care = mortality_in_care.append(idu_all_female_df.assign(group='idu_white_female'))
mortality_in_care = mortality_in_care.append(idu_all_female_df.assign(group='idu_hisp_female'))
mortality_in_care = mortality_in_care.set_index('group').sort_index().reset_index()

# Clean out care coefficients
mortality_out_care = pd.read_csv(f'{in_dir}/mortality_coeff_out_of_care.csv')
mortality_out_care.columns = mortality_out_care.columns.str.lower()
mortality_out_care['sex'] = np.where(mortality_out_care['sex'] ==1, 'male', 'female')
mortality_out_care['pop2'] = mortality_out_care['pop2'].str.lower()
mortality_out_care['group'] = mortality_out_care['pop2'] + '_' + mortality_out_care['sex']
mortality_out_care = mortality_out_care[['group', 'intercept_c', 'year_c', 'agecat_c', 'tv_sqrtcd4n_c', 'smoking_c', 'hcv_c', 'anx_c', 'dpr_c', 'ht_c', 'dm_c', 'ckd_c', 'lipid_c']].copy()
idu_all_female = mortality_out_care['group'] == 'idu_all_female'
idu_all_female_df = mortality_out_care.loc[idu_all_female].copy()
mortality_out_care = mortality_out_care.loc[~idu_all_female]
mortality_out_care = mortality_out_care.append(idu_all_female_df.assign(group='idu_black_female'))
mortality_out_care = mortality_out_care.append(idu_all_female_df.assign(group='idu_white_female'))
mortality_out_care = mortality_out_care.append(idu_all_female_df.assign(group='idu_hisp_female'))
mortality_out_care = mortality_out_care.set_index('group').sort_index().reset_index()

# Save them
mortality_in_care.to_feather(f'{out_dir}/mortality_in_care.feather')
mortality_out_care.to_feather(f'{out_dir}/mortality_out_care.feather')
