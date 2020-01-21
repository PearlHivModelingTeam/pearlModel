# Imports
import os
import numpy as np
import pandas as pd
import feather

pd.set_option("display.max_rows", 1001)

# Define directories
cwd = os.getcwd()
param_dir = cwd + '/../../data/parameters'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

def clean_coeff(df):
    """Format tables holding coefficient values"""
    df.columns = df.columns.str.lower()

    # Combine sex risk and race into single group indentifier
    if (np.issubdtype(df.sex.dtype, np.number)):
        df.sex = np.where(df.sex == 1, 'male', 'female')
    else:
        df.sex = np.where(df.sex == 'Males', 'male', 'female')
    if ('pop2' in df.columns):
        df = df.rename(columns={'pop2':'group'})
        df.group = df.group.str.decode('utf-8')
    else:
        df.group = df.group.astype('str')

    df.group = df.group + '_' + df.sex
    df.group = df.group.str.lower()
    df = df.drop(columns='sex')
    df = df.sort_values(by='group')
    df = df.set_index('group')
    return(df)

# Number of people on art in 2009: on_art_2009
on_art_2009 = feather.read_dataframe(f'{param_dir}/on_art_2009.feather').set_index(['group']).sort_index()

# Proportion of people with certain h1yy given age, risk, sex: h1yy_by_age_2009
h1yy_by_age_2009 = feather.read_dataframe(f'{param_dir}/h1yy_by_age_2009.feather').set_index(['group', 'age2009cat', 'h1yy']).sort_index()

# Mean and std of sqrtcd4n as a glm of h1yy for each group in 2009: cd4n_by_h1yy_2009
cd4n_by_h1yy_2009 = feather.read_dataframe(f'{param_dir}/cd4n_by_h1yy_2009.feather').set_index(['group']).sort_index()

# Mixed gaussian coefficients for age of patients alive in 2009: age_in_2009
age_in_2009 = feather.read_dataframe(f'{param_dir}/age_in_2009.feather')
neg = age_in_2009.loc[(age_in_2009['term'] == 'lambda1') & (age_in_2009['conf_low'] < 0.0)].copy()
neg['conf_low'] = 0.0
neg['conf_high'] = 2.0 * neg['estimate']
age_in_2009.loc[(age_in_2009['term'] == 'lambda1') & (age_in_2009['conf_low'] < 0.0)] = neg
pos = age_in_2009.loc[(age_in_2009['term'] == 'lambda1') & (age_in_2009['conf_high'] > 1.0)].copy()
pos['conf_high'] = 1.0
pos['conf_low'] = 1.0 - (2.0 * (1.0 - pos['estimate']))
age_in_2009.loc[(age_in_2009['term'] == 'lambda1') & (age_in_2009['conf_high'] > 1.0)] = pos
age_in_2009 = age_in_2009.set_index(['group', 'term']).sort_index(level=0)
age_in_2009.loc[('idu_hisp_female', 'lambda1'), :] = 0.0

# New dx and dx prediction intervals
new_dx = feather.read_dataframe(f'{param_dir}/new_dx.feather').set_index(['group', 'year'])
new_dx_interval = feather.read_dataframe(f'{param_dir}/new_dx_interval.feather').set_index(['group', 'year'])

# Age at haart init mixed gaussian coefficients
age_by_h1yy = feather.read_dataframe(f'{param_dir}/age_by_h1yy.feather').set_index(['group', 'param', 'h1yy']).sort_index()

# Mean and std of sqrtcd4n as a glm of h1yy for each group: cd4n_by_h1yy
cd4n_by_h1yy = feather.read_dataframe(f'{param_dir}/cd4n_by_h1yy.feather').set_index('group').sort_index()

# Coefficients for mortality in care
mortality_in_care = feather.read_dataframe(f'{param_dir}/mortality_in_care.feather')
cols = mortality_in_care.columns.tolist()
mortality_in_care = mortality_in_care.set_index('group')
mortality_in_care_vcov = feather.read_dataframe(f'{param_dir}/mortality_in_care_vcov.feather')
mortality_in_care_vcov.columns = cols
mortality_in_care_vcov['covariate'] = 15*(cols[1:])
mortality_in_care_vcov = mortality_in_care_vcov.set_index(['group', 'covariate'])


# Coefficients for mortality out of care
mortality_out_care = feather.read_dataframe(f'{param_dir}/mortality_out_care.feather')
cols = mortality_out_care.columns.tolist()
mortality_out_care = mortality_out_care.set_index('group')
mortality_out_care_vcov = feather.read_dataframe(f'{param_dir}/mortality_out_care_vcov.feather')
mortality_out_care_vcov.columns = cols
mortality_out_care_vcov['covariate'] = 15*(cols[1:])
mortality_out_care_vcov = mortality_out_care_vcov.set_index(['group', 'covariate'])

# Coefficients for loss to follow up
loss_to_follow_up = feather.read_dataframe(f'{param_dir}/loss_to_follow_up.feather')
cols = loss_to_follow_up.columns.tolist()
loss_to_follow_up = loss_to_follow_up.set_index('group')
loss_to_follow_up_vcov = feather.read_dataframe(f'{param_dir}/loss_to_follow_up_vcov.feather')
loss_to_follow_up_vcov.columns = cols
loss_to_follow_up_vcov['covariate'] = 15*(cols[1:])
loss_to_follow_up_vcov = loss_to_follow_up_vcov.set_index(['group', 'covariate'])
ltfu_knots = clean_coeff(pd.read_sas(param_dir + '/ltfu_knots.sas7bdat'))

# Coefficients for cd4 decline out of care
cd4_decrease = feather.read_dataframe(f'{param_dir}/cd4_decrease.feather')
cols = cd4_decrease.columns.tolist()
cd4_decrease['group'] = 'all'
cd4_decrease = cd4_decrease.set_index('group')
cd4_decrease_vcov = feather.read_dataframe(f'{param_dir}/cd4_decrease_vcov.feather')
cd4_decrease_vcov.columns = cols

# Coefficients of cd4 increase over time
cd4_increase = feather.read_dataframe(f'{param_dir}/cd4_increase.feather')
cols = cd4_increase.columns.tolist()
cd4_increase = cd4_increase.set_index('group')
cd4_increase_vcov = feather.read_dataframe(f'{param_dir}/cd4_increase_vcov.feather')
cd4_increase_vcov.columns = cols
cd4_increase_vcov['covariate'] = 15*(cols[1:])
cd4_increase_vcov = cd4_increase_vcov.set_index(['group', 'covariate'])
cd4_increase_knots = pd.DataFrame({'group': group_names, 'p5': 15*[1.0], 'p35': 15*[3.0], 'p65': 15*[6.0], 'p95': 15*[12.0]}).set_index('group')

# Probability to reengage in care for each group
prob_reengage = pd.read_csv(f'{param_dir}/prob_reengage.csv').set_index(['group'])

# Stage 0 comorbidities
hcv_prev_users = pd.read_feather(f'{param_dir}/stage_0/hcv_prev_users.feather').set_index('group')
hcv_prev_inits = pd.read_feather(f'{param_dir}/stage_0/hcv_prev_inits.feather').set_index('group')
smoking_prev_users = pd.read_feather(f'{param_dir}/stage_0/smoking_prev_users.feather').set_index('group')
smoking_prev_inits = pd.read_feather(f'{param_dir}/stage_0/smoking_prev_inits.feather').set_index('group')

# Stage 1 comorbidities
anxiety_prev_users = pd.read_feather(f'{param_dir}/stage_1/anxiety_prev_users.feather').set_index('group')
depression_prev_users = pd.read_feather(f'{param_dir}/stage_1/depression_prev_users.feather').set_index('group')
anxiety_prev_inits = pd.read_feather(f'{param_dir}/stage_1/anxiety_prev_inits.feather').set_index('group')
depression_prev_inits = pd.read_feather(f'{param_dir}/stage_1/depression_prev_inits.feather').set_index('group')
anxiety_coeff = pd.read_feather(f'{param_dir}/stage_1/anxiety_coeff.feather').set_index(['group', 'param']).unstack()
depression_coeff = pd.read_feather(f'{param_dir}/stage_1/depression_coeff.feather').set_index(['group', 'param']).unstack()

# Stage 2 comorbidities
ckd_prev_users = pd.read_feather(f'{param_dir}/stage_2/ckd_prev_users.feather').set_index('group')
lipid_prev_users = pd.read_feather(f'{param_dir}/stage_2/lipid_prev_users.feather').set_index('group')
diabetes_prev_users = pd.read_feather(f'{param_dir}/stage_2/diabetes_prev_users.feather').set_index('group')
hypertension_prev_users = pd.read_feather(f'{param_dir}/stage_2/hypertension_prev_users.feather').set_index('group')

ckd_prev_inits = pd.read_feather(f'{param_dir}/stage_2/ckd_prev_inits.feather').set_index('group')
lipid_prev_inits = pd.read_feather(f'{param_dir}/stage_2/lipid_prev_inits.feather').set_index('group')
diabetes_prev_inits = pd.read_feather(f'{param_dir}/stage_2/diabetes_prev_inits.feather').set_index('group')
hypertension_prev_inits = pd.read_feather(f'{param_dir}/stage_2/hypertension_prev_inits.feather').set_index('group')

ckd_coeff = pd.read_feather(f'{param_dir}/stage_2/ckd_coeff.feather').set_index(['group', 'param']).unstack()
lipid_coeff = pd.read_feather(f'{param_dir}/stage_2/lipid_coeff.feather').set_index(['group', 'param']).unstack()
diabetes_coeff = pd.read_feather(f'{param_dir}/stage_2/diabetes_coeff.feather').set_index(['group', 'param']).unstack()
hypertension_coeff = pd.read_feather(f'{param_dir}/stage_2/hypertension_coeff.feather').set_index(['group', 'param']).unstack()

# Comortality
mortality_in_care_co = pd.read_feather(f'{param_dir}/comortality/mortality_in_care.feather').set_index('group')
mortality_out_care_co = pd.read_feather(f'{param_dir}/comortality/mortality_out_care.feather').set_index('group')

# Save everything
with pd.HDFStore(param_dir + '/parameters.h5') as store:
    store['on_art_2009'] = on_art_2009
    store['h1yy_by_age_2009'] = h1yy_by_age_2009 
    store['cd4n_by_h1yy_2009'] = cd4n_by_h1yy_2009
    store['age_in_2009'] = age_in_2009
    store['new_dx'] = new_dx
    store['new_dx_interval'] = new_dx_interval
    store['age_by_h1yy'] = age_by_h1yy
    store['cd4n_by_h1yy'] = cd4n_by_h1yy
    store['mortality_in_care'] = mortality_in_care
    store['mortality_in_care_vcov'] = mortality_in_care_vcov
    store['mortality_out_care'] = mortality_out_care
    store['mortality_out_care_vcov'] = mortality_out_care_vcov
    store['loss_to_follow_up'] = loss_to_follow_up
    store['loss_to_follow_up_vcov'] = loss_to_follow_up_vcov
    store['ltfu_knots'] = ltfu_knots
    store['cd4_decrease'] = cd4_decrease
    store['cd4_decrease_vcov'] = cd4_decrease_vcov
    store['cd4_increase'] = cd4_increase
    store['cd4_increase_vcov'] = cd4_increase_vcov
    store['cd4_increase_knots'] = cd4_increase_knots
    store['prob_reengage'] = prob_reengage

    # Stage 0 comorbidities
    store['hcv_prev_users'] = hcv_prev_users
    store['hcv_prev_inits'] = hcv_prev_inits
    store['smoking_prev_users'] = smoking_prev_users
    store['smoking_prev_inits'] = smoking_prev_inits

    # Stage 1 comorbidities
    store['anxiety_prev_users'] = anxiety_prev_users
    store['anxiety_prev_inits'] = anxiety_prev_inits
    store['anxiety_coeff'] = anxiety_coeff
    store['depression_prev_users'] = depression_prev_users
    store['depression_prev_inits'] = depression_prev_inits
    store['depression_coeff'] = depression_coeff

    # Stage 2 comorbidities
    store['ckd_prev_users'] = ckd_prev_users
    store['lipid_prev_users'] = lipid_prev_users
    store['diabetes_prev_users'] = diabetes_prev_users
    store['hypertension_prev_users'] = hypertension_prev_users

    store['ckd_prev_inits'] = ckd_prev_inits
    store['lipid_prev_inits'] = lipid_prev_inits
    store['diabetes_prev_inits'] = diabetes_prev_inits
    store['hypertension_prev_inits'] = hypertension_prev_inits

    store['ckd_coeff'] = ckd_coeff
    store['lipid_coeff'] = lipid_coeff
    store['diabetes_coeff'] = diabetes_coeff
    store['hypertension_coeff'] = hypertension_coeff

    # Comortality
    store['mortality_in_care_co'] = mortality_in_care_co
    store['mortality_out_care_co'] = mortality_out_care_co


