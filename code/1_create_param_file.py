# Imports
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Define directories
param_dir = Path('../param_files')

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']


def clean_coeff(df):
    """Format tables holding coefficient values"""
    df.columns = df.columns.str.lower()

    # Combine sex risk and race into single group identifier
    if np.issubdtype(df.sex.dtype, np.number):
        df.sex = np.where(df.sex == 1, 'male', 'female')
    else:
        df.sex = np.where(df.sex == 'Males', 'male', 'female')
    if 'pop2' in df.columns:
        df = df.rename(columns={'pop2':'group'})
        df.group = df.group.str.decode('utf-8')
    else:
        df.group = df.group.astype('str')

    df.group = df.group + '_' + df.sex
    df.group = df.group.str.lower()
    df = df.drop(columns='sex')
    df = df.sort_values(by='group')
    df = df.set_index('group')
    return df


# Number of people on art in 2009: on_art_2009
on_art_2009 = pd.read_csv(param_dir/'on_art_2009.csv').set_index(['group']).sort_index()

# Proportion of people with certain h1yy given age, risk, sex: h1yy_by_age_2009
h1yy_by_age_2009 = pd.read_csv(param_dir/'h1yy_by_age_2009.csv').set_index(['group', 'age2009cat', 'h1yy']).sort_index()[['pct']]
# If there is no data for a specific age group, then use the values for msm_white_male
h1yy_by_age_2009 = h1yy_by_age_2009.reindex(pd.MultiIndex.from_product([h1yy_by_age_2009.index.levels[0].unique(),
                                                                        h1yy_by_age_2009.index.levels[1].unique(),
                                                                        h1yy_by_age_2009.index.levels[2].unique()],
                                            names=['group', 'age_cat', 'h1yy']))
for group in h1yy_by_age_2009.index.levels[0].unique():
    for age_cat in h1yy_by_age_2009.index.levels[1].unique():
        if h1yy_by_age_2009.loc[group, age_cat].isnull().values.any():
            h1yy_by_age_2009.loc[group, age_cat] = h1yy_by_age_2009.loc['msm_white_male', age_cat].values

# Mean and std of sqrtcd4n as a glm of h1yy for each group in 2009
cd4n_by_h1yy_2009 = pd.read_csv(param_dir/'cd4n_by_h1yy_2009.csv').set_index(['group']).sort_index()
years = np.arange(2000, 2010)

# Unpack values for each year
df = pd.DataFrame(index=pd.MultiIndex.from_product([group_names, ['mu', 'sigma'], years], names=['group', 'param', 'h1yy']), columns=['estimate']).sort_index()
for group in group_names:
    mu = pd.DataFrame({'h1yy': years, 'estimate': cd4n_by_h1yy_2009.loc[group, 'meanint'] + cd4n_by_h1yy_2009.loc[group, 'meanslp'] * years})
    mu = mu.set_index('h1yy')
    sigma = pd.DataFrame({'h1yy': years, 'estimate': cd4n_by_h1yy_2009.loc[group, 'stdint'] + cd4n_by_h1yy_2009.loc[group, 'stdslp'] * years})
    sigma = sigma.set_index('h1yy')
    df.loc[(group, 'mu'), 'estimate'] = mu['estimate'].to_numpy()
    df.loc[(group, 'sigma'), 'estimate'] = sigma['estimate'].to_numpy()
df = df.reset_index().sort_values(['group', 'h1yy', 'param']).set_index(['group', 'h1yy', 'param'])
df['estimate'] = df['estimate'].astype(float)
cd4n_by_h1yy_2009 = df

# Mixed gaussian coefficients for age of patients alive in 2009
age_in_2009 = pd.read_csv(param_dir/'age_in_2009.csv')
# Truncate all numeric values at 0 and all lambda1 values at 1
for col in age_in_2009.select_dtypes(include=np.number).columns.tolist():
    age_in_2009.loc[age_in_2009[col] < 0, col] = 0
    age_in_2009.loc[(age_in_2009['term'] == 'lambda1') & (age_in_2009[col] > 1), col] = 1
age_in_2009 = age_in_2009.set_index(['group', 'term']).sort_index(level=0)

# New dx prediction intervals
new_dx = pd.read_csv(param_dir/'new_dx_interval.csv').set_index(['group', 'year'])
new_dx_ehe = pd.read_csv(param_dir/'new_dx_combined_ehe.csv').set_index(['group', 'year'])
new_dx_sa = pd.read_csv(param_dir/'new_dx_interval_sa.csv').set_index(['group', 'year'])

# Linkage to care
linkage_to_care = pd.read_csv(param_dir/'linkage_to_care.csv').set_index(['group', 'year'])

# Age at art init mixed gaussian coefficients
age_by_h1yy = pd.read_csv(param_dir/'age_by_h1yy.csv')
age_by_h1yy = age_by_h1yy.loc[(age_by_h1yy['param'] != 'lambda2') & (age_by_h1yy['h1yy'] != 2009)]
# No values less than 0 and no lambda1 greater than 1
age_by_h1yy.loc[age_by_h1yy['pred'] < 0, 'pred'] = 0
age_by_h1yy.loc[(age_by_h1yy['param'] == 'lambda1') & (age_by_h1yy['pred'] > 1), 'pred'] = 1.0
# Create column of 2018 values
values_2018 = age_by_h1yy.loc[age_by_h1yy['h1yy'] == 2018]['pred'].to_numpy()
total_years = len(age_by_h1yy['h1yy'].unique())
age_by_h1yy['value_2018'] = np.array([total_years*[value_2018] for value_2018 in values_2018]).flatten()
# Create range after 2018
age_by_h1yy['le_2018'] = (age_by_h1yy['h1yy'] <= 2018)
age_by_h1yy['pred2'] = age_by_h1yy['le_2018'].astype(int) * age_by_h1yy['pred'] + (~age_by_h1yy['le_2018']).astype(int) * age_by_h1yy['value_2018']
age_by_h1yy['low_value'] = age_by_h1yy[['pred', 'pred2']].min(axis=1)
age_by_h1yy['high_value'] = age_by_h1yy[['pred', 'pred2']].max(axis=1)
age_by_h1yy = age_by_h1yy[['group', 'param', 'h1yy', 'low_value', 'high_value']].sort_values(['group', 'h1yy', 'param']).set_index(['group', 'h1yy', 'param'])

# Mean and std of sqrtcd4n as a glm of h1yy for each group: cd4n_by_h1yy
cd4n_by_h1yy = pd.read_csv(param_dir/'cd4n_by_h1yy.csv').set_index('group').sort_index()
years = np.arange(2010, 2035)
params = ['mu', 'sigma']

df = pd.DataFrame(index=pd.MultiIndex.from_product([group_names, params, years], names=['group', 'param', 'h1yy']), columns=['low_value', 'high_value']).sort_index()
for group in group_names:
    # Create a range between the 2018 value and the predicted value
    mu = pd.DataFrame({'h1yy': years, 'low_value': cd4n_by_h1yy.loc[group, 'meanint'] + cd4n_by_h1yy.loc[group, 'meanslp'] * years})
    mu['high_value'] = mu['low_value']
    mu_2018 = mu.loc[mu['h1yy'] == 2018, 'high_value'].to_numpy()[0]
    mu.loc[mu['h1yy'] >= 2018, 'low_value'] = mu_2018
    mu = mu.set_index('h1yy')
    sigma = pd.DataFrame({'h1yy': years, 'low_value': cd4n_by_h1yy.loc[group, 'stdint'] + cd4n_by_h1yy.loc[group, 'stdslp'] * years})
    sigma['high_value'] = sigma['low_value']
    sigma_2018 = sigma.loc[sigma['h1yy']==2018, 'high_value'].to_numpy()[0]
    sigma.loc[sigma['h1yy'] >=2018, 'low_value'] = sigma_2018
    sigma = sigma.set_index('h1yy')

    # Save subpopulation dataframe to overall dataframe
    df.loc[(group, 'mu'), 'low_value'] = mu['low_value'].to_numpy()
    df.loc[(group, 'mu'), 'high_value'] = mu['high_value'].to_numpy()
    df.loc[(group, 'sigma'), 'low_value'] = sigma['low_value'].to_numpy()
    df.loc[(group, 'sigma'), 'high_value'] = sigma['high_value'].to_numpy()

df = df.reset_index().sort_values(['group', 'h1yy', 'param']).set_index(['group', 'h1yy', 'param'])
df['low_value'] = df['low_value'].astype(float)
df['high_value'] = df['high_value'].astype(float)
cd4n_by_h1yy = df

# Coefficients for mortality in care
mortality_in_care = pd.read_csv(param_dir/'mortality_in_care.csv')
cols = mortality_in_care.columns.tolist()
mortality_in_care = mortality_in_care.set_index('group')
mortality_in_care_age = pd.read_csv(param_dir/'mortality_in_care_age.csv').set_index('group')
mortality_in_care_sqrtcd4 = pd.read_csv(param_dir/'mortality_in_care_sqrtcd4.csv').set_index('group')
mortality_in_care_vcov = pd.read_csv(param_dir/'mortality_in_care_vcov.csv').set_index(['group', 'covariate'])

mortality_threshold = pd.read_csv(param_dir/'cdc_mortality.csv').set_index(['group', 'mortality_age_group'])


# Coefficients for mortality out of care
mortality_out_care = pd.read_csv(param_dir/'mortality_out_care.csv')
cols = mortality_out_care.columns.tolist()
mortality_out_care = mortality_out_care.set_index('group')
mortality_out_care_age = pd.read_csv(param_dir/'mortality_out_care_age.csv').set_index('group')
mortality_out_care_tv_sqrtcd4 = pd.read_csv(param_dir/'mortality_out_care_tv_sqrtcd4.csv').set_index('group')
mortality_out_care_vcov = pd.read_csv(param_dir/'mortality_out_care_vcov.csv').set_index(['group', 'covariate'])

# Coefficients for loss to follow up
loss_to_follow_up = pd.read_csv(param_dir/'loss_to_follow_up.csv')
cols = loss_to_follow_up.columns.tolist()
loss_to_follow_up = loss_to_follow_up.set_index('group')
loss_to_follow_up_vcov = pd.read_csv(param_dir/'loss_to_follow_up_vcov.csv')
loss_to_follow_up_vcov.columns = cols
loss_to_follow_up_vcov['covariate'] = 15*(cols[1:])
loss_to_follow_up_vcov = loss_to_follow_up_vcov.set_index(['group', 'covariate'])

ltfu_knots = pd.read_csv(param_dir/'ltfu_knots.csv').set_index('group')



# Coefficients for cd4 decline out of care
cd4_decrease = pd.read_csv(param_dir/'cd4_decrease.csv')
cols = cd4_decrease.columns.tolist()
cd4_decrease['group'] = 'all'
cd4_decrease = cd4_decrease.set_index('group')
cd4_decrease_vcov = pd.read_csv(param_dir/'cd4_decrease_vcov.csv')
cd4_decrease_vcov.columns = cols

# Coefficients and knots of cd4 increase over time
cd4_increase = pd.read_csv(param_dir/'cd4_increase.csv')
cols = cd4_increase.columns.tolist()
cd4_increase = cd4_increase.set_index('group')
cd4_increase_vcov = pd.read_csv(param_dir/'cd4_increase_vcov.csv')
cd4_increase_vcov.columns = cols
cd4_increase_vcov['covariate'] = 15*(cols[1:])
cd4_increase_vcov = cd4_increase_vcov.set_index(['group', 'covariate'])
cd4_increase_knots = pd.DataFrame({'group': group_names, 'p5': 15*[1.0], 'p35': 15*[4.0], 'p65': 15*[7.0], 'p95': 15*[13.0]}).set_index('group')

# Number of years spent out of care
years_out_of_care = pd.read_csv(param_dir/'years_out_of_care.csv')

# BMI
pre_art_bmi = pd.read_csv(param_dir/'aim2/bmi/pre_art_bmi.csv').set_index(['group', 'parameter'])
pre_art_bmi_model = pd.read_csv(param_dir/'aim2/bmi/pre_art_bmi_model.csv').set_index(['group'])
pre_art_bmi_age_knots = pd.read_csv(param_dir/'aim2/bmi/pre_art_bmi_age_knots.csv').set_index('group')
pre_art_bmi_h1yy_knots = pd.read_csv(param_dir/'aim2/bmi/pre_art_bmi_h1yy_knots.csv').set_index('group')
post_art_bmi = pd.read_csv(param_dir/'aim2/bmi/post_art_bmi.csv').set_index(['group', 'parameter'])
post_art_bmi_age_knots = pd.read_csv(param_dir/'aim2/bmi/post_art_bmi_age_knots.csv').set_index(['group'])
post_art_bmi_pre_art_bmi_knots = pd.read_csv(param_dir/'aim2/bmi/post_art_bmi_pre_art_bmi_knots.csv').set_index(['group'])
post_art_bmi_cd4_knots = pd.read_csv(param_dir/'aim2/bmi/post_art_bmi_cd4_knots.csv').set_index(['group'])
post_art_bmi_cd4_post_knots = pd.read_csv(param_dir/'aim2/bmi/post_art_bmi_cd4_post_knots.csv').set_index(['group'])

# Stage 0 comorbidities
hcv_prev_users = pd.read_csv(param_dir/'aim2/stage0/hcv_prev_users.csv').set_index('group')
hcv_prev_inits = pd.read_csv(param_dir/'aim2/stage0/hcv_prev_inits.csv').set_index('group')
smoking_prev_users = pd.read_csv(param_dir/'aim2/stage0/smoking_prev_users.csv').set_index('group')
smoking_prev_inits = pd.read_csv(param_dir/'aim2/stage0/smoking_prev_inits.csv').set_index('group')

# Stage 1 comorbidities
anx_prev_users = pd.read_csv(param_dir/'aim2/stage1/anxiety_prev_users.csv').set_index('group')
dpr_prev_users = pd.read_csv(param_dir/'aim2/stage1/depression_prev_users.csv').set_index('group')
anx_prev_inits = pd.read_csv(param_dir/'aim2/stage1/anxiety_prev_inits.csv').set_index('group')
dpr_prev_inits = pd.read_csv(param_dir/'aim2/stage1/depression_prev_inits.csv').set_index('group')
anx_coeff = pd.read_csv(param_dir/'aim2/stage1/anxiety_coeff.csv').set_index(['group', 'param']).unstack()
dpr_coeff = pd.read_csv(param_dir/'aim2/stage1/depression_coeff.csv').set_index(['group', 'param']).unstack()

# Stage 2 comorbidities
ckd_prev_users = pd.read_csv(param_dir/'aim2/stage2/ckd_prev_users.csv').set_index('group')
lipid_prev_users = pd.read_csv(param_dir/'aim2/stage2/lipid_prev_users.csv').set_index('group')
dm_prev_users = pd.read_csv(param_dir/'aim2/stage2/dm_prev_users.csv').set_index('group')
ht_prev_users = pd.read_csv(param_dir/'aim2/stage2/ht_prev_users.csv').set_index('group')

ckd_prev_inits = pd.read_csv(param_dir/'aim2/stage2/ckd_prev_inits.csv').set_index('group')
lipid_prev_inits = pd.read_csv(param_dir/'aim2/stage2/lipid_prev_inits.csv').set_index('group')
dm_prev_inits = pd.read_csv(param_dir/'aim2/stage2/dm_prev_inits.csv').set_index('group')
ht_prev_inits = pd.read_csv(param_dir/'aim2/stage2/ht_prev_inits.csv').set_index('group')

ckd_coeff = pd.read_csv(param_dir/'aim2/stage2/ckd_coeff.csv').set_index(['group', 'param']).unstack()
lipid_coeff = pd.read_csv(param_dir/'aim2/stage2/lipid_coeff.csv').set_index(['group', 'param']).unstack()
dm_coeff = pd.read_csv(param_dir/'aim2/stage2/dm_coeff.csv').set_index(['group', 'param']).unstack()
ht_coeff = pd.read_csv(param_dir/'aim2/stage2/ht_coeff.csv').set_index(['group', 'param']).unstack()

ckd_delta_bmi = pd.read_csv(param_dir/'aim2/stage2/ckd_delta_bmi.csv').set_index('group')
dm_delta_bmi = pd.read_csv(param_dir/'aim2/stage2/dm_delta_bmi.csv').set_index('group')
ht_delta_bmi = pd.read_csv(param_dir/'aim2/stage2/ht_delta_bmi.csv').set_index('group')
lipid_delta_bmi = pd.read_csv(param_dir/'aim2/stage2/lipid_delta_bmi.csv').set_index('group')

ckd_post_art_bmi = pd.read_csv(param_dir/'aim2/stage2/ckd_post_art_bmi.csv').set_index('group')
dm_post_art_bmi = pd.read_csv(param_dir/'aim2/stage2/dm_post_art_bmi.csv').set_index('group')
ht_post_art_bmi = pd.read_csv(param_dir/'aim2/stage2/ht_post_art_bmi.csv').set_index('group')
lipid_post_art_bmi = pd.read_csv(param_dir/'aim2/stage2/lipid_post_art_bmi.csv').set_index('group')

# Stage 3 comorbidities
malig_prev_users = pd.read_csv(param_dir/'aim2/stage3/malig_prev_users.csv').set_index('group')
esld_prev_users = pd.read_csv(param_dir/'aim2/stage3/esld_prev_users.csv').set_index('group')
mi_prev_users = pd.read_csv(param_dir/'aim2/stage3/mi_prev_users.csv').set_index('group')

malig_prev_ini = pd.read_csv(param_dir/'aim2/stage3/malig_prev_ini.csv').set_index('group')
esld_prev_ini = pd.read_csv(param_dir/'aim2/stage3/esld_prev_ini.csv').set_index('group')
mi_prev_ini = pd.read_csv(param_dir/'aim2/stage3/mi_prev_ini.csv').set_index('group')

malig_coeff = pd.read_csv(param_dir/'aim2/stage3/malig_coeff.csv').set_index(['group', 'param']).unstack()
esld_coeff = pd.read_csv(param_dir/'aim2/stage3/esld_coeff.csv').set_index(['group', 'param']).unstack()
mi_coeff = pd.read_csv(param_dir/'aim2/stage3/mi_coeff.csv').set_index(['group', 'param']).unstack()

malig_delta_bmi = pd.read_csv(param_dir/'aim2/stage3/malig_delta_bmi.csv').set_index('group')
esld_delta_bmi = pd.read_csv(param_dir/'aim2/stage3/esld_delta_bmi.csv').set_index('group')
mi_delta_bmi = pd.read_csv(param_dir/'aim2/stage3/mi_delta_bmi.csv').set_index('group')

malig_post_art_bmi = pd.read_csv(param_dir/'aim2/stage3/malig_post_art_bmi.csv').set_index('group')
esld_post_art_bmi = pd.read_csv(param_dir/'aim2/stage3/esld_post_art_bmi.csv').set_index('group')
mi_post_art_bmi = pd.read_csv(param_dir/'aim2/stage3/mi_post_art_bmi.csv').set_index('group')

# mortality with comorbidity
mortality_in_care_co = pd.read_csv(param_dir/'aim2/mortality/mortality_in_care.csv').set_index(['group', 'param']).unstack()
mortality_in_care_post_art_bmi = pd.read_csv(param_dir/'aim2/mortality/mortality_in_care_post_art_bmi.csv').set_index('group')

mortality_out_care_co = pd.read_csv(param_dir/'aim2/mortality/mortality_out_care.csv').set_index(['group', 'param']).unstack()
mortality_out_care_post_art_bmi = pd.read_csv(param_dir/'aim2/mortality/mortality_out_care_post_art_bmi.csv').set_index('group')

# Mortality paper options
mortality_in_care_overall = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_overall.csv').set_index('group')
mortality_in_care_age_overall = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_age_overall.csv').set_index('group')
mortality_in_care_sqrtcd4_overall = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_sqrtcd4_overall.csv').set_index('group')
mortality_out_care_overall = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_overall.csv').set_index('group')
mortality_out_care_age_overall = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_age_overall.csv').set_index('group')
mortality_out_care_tv_sqrtcd4_overall = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_tv_sqrtcd4_overall.csv').set_index('group')
mortality_threshold_overall = pd.read_csv(param_dir/'mortality_paper/cdc_mortality_overall.csv').set_index(['group', 'mortality_age_group'])

mortality_in_care_by_sex = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_by_sex.csv').set_index('group')
mortality_in_care_age_by_sex = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_age_by_sex.csv').set_index('group')
mortality_in_care_sqrtcd4_by_sex = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_sqrtcd4_by_sex.csv').set_index('group')
mortality_out_care_by_sex = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_by_sex.csv').set_index('group')
mortality_out_care_age_by_sex = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_age_by_sex.csv').set_index('group')
mortality_out_care_tv_sqrtcd4_by_sex = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_tv_sqrtcd4_by_sex.csv').set_index('group')
mortality_threshold_by_sex = pd.read_csv(param_dir/'mortality_paper/cdc_mortality_by_sex.csv').set_index(['group', 'mortality_age_group'])

mortality_in_care_by_sex_race = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_by_sex_race.csv').set_index('group')
mortality_in_care_age_by_sex_race = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_age_by_sex_race.csv').set_index('group')
mortality_in_care_sqrtcd4_by_sex_race = pd.read_csv(param_dir/'mortality_paper/mortality_in_care/mortality_in_care_sqrtcd4_by_sex_race.csv').set_index('group')
mortality_out_care_by_sex_race = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_by_sex_race.csv').set_index('group')
mortality_out_care_age_by_sex_race = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_age_by_sex_race.csv').set_index('group')
mortality_out_care_tv_sqrtcd4_by_sex_race = pd.read_csv(param_dir/'mortality_paper/mortality_out_care/mortality_out_care_tv_sqrtcd4_by_sex_race.csv').set_index('group')
mortality_threshold_by_sex_race = pd.read_csv(param_dir/'mortality_paper/cdc_mortality_by_sex_race.csv').set_index(['group', 'mortality_age_group'])

# Mortality paper 2015
mortality_in_care_overall_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_overall_2015.csv').set_index('group')
mortality_in_care_age_overall_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_age_overall_2015.csv').set_index('group')
mortality_in_care_sqrtcd4_overall_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_sqrtcd4_overall_2015.csv').set_index('group')
mortality_out_care_overall_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_overall_2015.csv').set_index('group')
mortality_out_care_age_overall_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_age_overall_2015.csv').set_index('group')
mortality_out_care_tv_sqrtcd4_overall_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_tv_sqrtcd4_overall_2015.csv').set_index('group')
mortality_threshold_overall_2015 = pd.read_csv(param_dir/'mortality_paper/cdc_mortality_overall.csv').set_index(['group', 'mortality_age_group'])

mortality_in_care_by_sex_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_by_sex_2015.csv').set_index('group')
mortality_in_care_age_by_sex_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_age_by_sex_2015.csv').set_index('group')
mortality_in_care_sqrtcd4_by_sex_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_sqrtcd4_by_sex_2015.csv').set_index('group')
mortality_out_care_by_sex_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_by_sex_2015.csv').set_index('group')
mortality_out_care_age_by_sex_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_age_by_sex_2015.csv').set_index('group')
mortality_out_care_tv_sqrtcd4_by_sex_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_tv_sqrtcd4_by_sex_2015.csv').set_index('group')
mortality_threshold_by_sex_2015 = pd.read_csv(param_dir/'mortality_paper/cdc_mortality_by_sex.csv').set_index(['group', 'mortality_age_group'])

mortality_in_care_by_sex_race_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_by_sex_race_2015.csv').set_index('group')
mortality_in_care_age_by_sex_race_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_age_by_sex_race_2015.csv').set_index('group')
mortality_in_care_sqrtcd4_by_sex_race_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_sqrtcd4_by_sex_race_2015.csv').set_index('group')
mortality_out_care_by_sex_race_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_by_sex_race_2015.csv').set_index('group')
mortality_out_care_age_by_sex_race_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_age_by_sex_race_2015.csv').set_index('group')
mortality_out_care_tv_sqrtcd4_by_sex_race_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_tv_sqrtcd4_by_sex_race_2015.csv').set_index('group')
mortality_threshold_by_sex_race_2015 = pd.read_csv(param_dir/'mortality_paper/cdc_mortality_by_sex_race.csv').set_index(['group', 'mortality_age_group'])

mortality_in_care_by_sex_race_risk_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_by_sex_race_risk_2015.csv').set_index('group')
mortality_in_care_age_by_sex_race_risk_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_age_by_sex_race_risk_2015.csv').set_index('group')
mortality_in_care_sqrtcd4_by_sex_race_risk_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_in_care_sqrtcd4_by_sex_race_risk_2015.csv').set_index('group')
mortality_out_care_by_sex_race_risk_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_by_sex_race_risk_2015.csv').set_index('group')
mortality_out_care_age_by_sex_race_risk_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_age_by_sex_race_risk_2015.csv').set_index('group')
mortality_out_care_tv_sqrtcd4_by_sex_race_risk_2015 = pd.read_csv(param_dir/'mortality_paper_sa/mortality_out_care_tv_sqrtcd4_by_sex_race_risk_2015.csv').set_index('group')
mortality_threshold_by_sex_race_risk_2015 = pd.read_csv(param_dir/'cdc_mortality.csv').set_index(['group', 'mortality_age_group'])

mortality_threshold_idu_5x = pd.read_csv(param_dir/'cdc_mortality_idu_5x.csv').set_index(['group', 'mortality_age_group'])
mortality_threshold_idu_10x = pd.read_csv(param_dir/'cdc_mortality_idu_10x.csv').set_index(['group', 'mortality_age_group'])

# Save everything
out_file = param_dir/'parameters.h5'
if out_file.is_file():
    os.remove(out_file)

with pd.HDFStore(out_file) as store:
    store['on_art_2009'] = on_art_2009
    store['h1yy_by_age_2009'] = h1yy_by_age_2009
    store['cd4n_by_h1yy_2009'] = cd4n_by_h1yy_2009
    store['age_in_2009'] = age_in_2009
    store['new_dx'] = new_dx
    store['new_dx_ehe'] = new_dx_ehe
    store['new_dx_sa'] = new_dx_sa
    store['linkage_to_care'] = linkage_to_care
    store['age_by_h1yy'] = age_by_h1yy
    store['cd4n_by_h1yy'] = cd4n_by_h1yy
    store['mortality_in_care'] = mortality_in_care
    store['mortality_in_care_age'] = mortality_in_care_age
    store['mortality_in_care_sqrtcd4'] = mortality_in_care_sqrtcd4
    store['mortality_threshold'] = mortality_threshold
    store['mortality_in_care_vcov'] = mortality_in_care_vcov
    store['mortality_out_care'] = mortality_out_care
    store['mortality_out_care_age'] = mortality_out_care_age
    store['mortality_out_care_tv_sqrtcd4'] = mortality_out_care_tv_sqrtcd4
    store['mortality_out_care_vcov'] = mortality_out_care_vcov
    store['loss_to_follow_up'] = loss_to_follow_up
    store['loss_to_follow_up_vcov'] = loss_to_follow_up_vcov
    store['ltfu_knots'] = ltfu_knots
    store['cd4_decrease'] = cd4_decrease
    store['cd4_decrease_vcov'] = cd4_decrease_vcov
    store['cd4_increase'] = cd4_increase
    store['cd4_increase_vcov'] = cd4_increase_vcov
    store['cd4_increase_knots'] = cd4_increase_knots
    store['years_out_of_care'] = years_out_of_care

    # BMI
    store['pre_art_bmi'] = pre_art_bmi
    store['pre_art_bmi_model'] = pre_art_bmi_model
    store['pre_art_bmi_age_knots'] = pre_art_bmi_age_knots
    store['pre_art_bmi_h1yy_knots'] = pre_art_bmi_h1yy_knots
    store['post_art_bmi'] = post_art_bmi
    store['post_art_bmi_age_knots'] = post_art_bmi_age_knots
    store['post_art_bmi_pre_art_bmi_knots'] = post_art_bmi_pre_art_bmi_knots
    store['post_art_bmi_cd4_knots'] = post_art_bmi_cd4_knots
    store['post_art_bmi_cd4_post_knots'] = post_art_bmi_cd4_post_knots

    # Stage 0 comorbidities
    store['hcv_prev_users'] = hcv_prev_users
    store['hcv_prev_inits'] = hcv_prev_inits
    store['smoking_prev_users'] = smoking_prev_users
    store['smoking_prev_inits'] = smoking_prev_inits

    # Stage 1 comorbidities
    store['anx_prev_users'] = anx_prev_users
    store['anx_prev_inits'] = anx_prev_inits
    store['anx_coeff'] = anx_coeff
    store['dpr_prev_users'] = dpr_prev_users
    store['dpr_prev_inits'] = dpr_prev_inits
    store['dpr_coeff'] = dpr_coeff

    # Stage 2 comorbidities
    store['ckd_prev_users'] = ckd_prev_users
    store['lipid_prev_users'] = lipid_prev_users
    store['dm_prev_users'] = dm_prev_users
    store['ht_prev_users'] = ht_prev_users
    store['ckd_prev_inits'] = ckd_prev_inits
    store['lipid_prev_inits'] = lipid_prev_inits
    store['dm_prev_inits'] = dm_prev_inits
    store['ht_prev_inits'] = ht_prev_inits
    store['ckd_coeff'] = ckd_coeff
    store['lipid_coeff'] = lipid_coeff
    store['dm_coeff'] = dm_coeff
    store['ht_coeff'] = ht_coeff
    store['ckd_delta_bmi'] = ckd_delta_bmi
    store['lipid_delta_bmi'] = lipid_delta_bmi
    store['dm_delta_bmi'] = dm_delta_bmi
    store['ht_delta_bmi'] = ht_delta_bmi
    store['ckd_post_art_bmi'] = ckd_post_art_bmi
    store['lipid_post_art_bmi'] = lipid_post_art_bmi
    store['dm_post_art_bmi'] = dm_post_art_bmi
    store['ht_post_art_bmi'] = ht_post_art_bmi

    # Stage 3 comorbidities
    store['malig_prev_users'] = malig_prev_users
    store['esld_prev_users'] = esld_prev_users
    store['mi_prev_users'] = mi_prev_users
    store['malig_prev_inits'] = malig_prev_ini
    store['esld_prev_inits'] = esld_prev_ini
    store['mi_prev_inits'] = mi_prev_ini
    store['malig_coeff'] = malig_coeff
    store['esld_coeff'] = esld_coeff
    store['mi_coeff'] = mi_coeff
    store['malig_delta_bmi'] = malig_delta_bmi
    store['esld_delta_bmi'] = esld_delta_bmi
    store['mi_delta_bmi'] = mi_delta_bmi
    store['malig_post_art_bmi'] = malig_post_art_bmi
    store['esld_post_art_bmi'] = esld_post_art_bmi
    store['mi_post_art_bmi'] = mi_post_art_bmi

    # Comorbidity modified mortality
    store['mortality_in_care_co'] = mortality_in_care_co
    store['mortality_in_care_post_art_bmi'] = mortality_in_care_post_art_bmi
    store['mortality_out_care_co'] = mortality_out_care_co
    store['mortality_out_care_post_art_bmi'] = mortality_out_care_post_art_bmi

    # Mortality Paper Additions
    store['mortality_in_care_overall'] = mortality_in_care_overall
    store['mortality_in_care_age_overall'] = mortality_in_care_age_overall
    store['mortality_in_care_sqrtcd4_overall'] = mortality_in_care_sqrtcd4_overall
    store['mortality_out_care_overall'] = mortality_out_care_overall
    store['mortality_out_care_age_overall'] = mortality_out_care_age_overall
    store['mortality_out_care_tv_sqrtcd4_overall'] = mortality_out_care_tv_sqrtcd4_overall
    store['mortality_threshold_overall'] = mortality_threshold_overall

    store['mortality_in_care_by_sex'] = mortality_in_care_by_sex
    store['mortality_in_care_age_by_sex'] = mortality_in_care_age_by_sex
    store['mortality_in_care_sqrtcd4_by_sex'] = mortality_in_care_sqrtcd4_by_sex
    store['mortality_out_care_by_sex'] = mortality_out_care_by_sex
    store['mortality_out_care_age_by_sex'] = mortality_out_care_age_by_sex
    store['mortality_out_care_tv_sqrtcd4_by_sex'] = mortality_out_care_tv_sqrtcd4_by_sex
    store['mortality_threshold_by_sex'] = mortality_threshold_by_sex

    store['mortality_in_care_by_sex_race'] = mortality_in_care_by_sex_race
    store['mortality_in_care_age_by_sex_race'] = mortality_in_care_age_by_sex_race
    store['mortality_in_care_sqrtcd4_by_sex_race'] = mortality_in_care_sqrtcd4_by_sex_race
    store['mortality_out_care_by_sex_race'] = mortality_out_care_by_sex_race
    store['mortality_out_care_age_by_sex_race'] = mortality_out_care_age_by_sex_race
    store['mortality_out_care_tv_sqrtcd4_by_sex_race'] = mortality_out_care_tv_sqrtcd4_by_sex_race
    store['mortality_threshold_by_sex_race'] = mortality_threshold_by_sex_race

    store['mortality_in_care_overall_2015'] = mortality_in_care_overall_2015
    store['mortality_in_care_age_overall_2015'] = mortality_in_care_age_overall_2015
    store['mortality_in_care_sqrtcd4_overall_2015'] = mortality_in_care_sqrtcd4_overall_2015
    store['mortality_out_care_overall_2015'] = mortality_out_care_overall_2015
    store['mortality_out_care_age_overall_2015'] = mortality_out_care_age_overall_2015
    store['mortality_out_care_tv_sqrtcd4_overall_2015'] = mortality_out_care_tv_sqrtcd4_overall_2015
    store['mortality_threshold_overall_2015'] = mortality_threshold_overall_2015

    store['mortality_in_care_by_sex_2015'] = mortality_in_care_by_sex_2015
    store['mortality_in_care_age_by_sex_2015'] = mortality_in_care_age_by_sex_2015
    store['mortality_in_care_sqrtcd4_by_sex_2015'] = mortality_in_care_sqrtcd4_by_sex_2015
    store['mortality_out_care_by_sex_2015'] = mortality_out_care_by_sex_2015
    store['mortality_out_care_age_by_sex_2015'] = mortality_out_care_age_by_sex_2015
    store['mortality_out_care_tv_sqrtcd4_by_sex_2015'] = mortality_out_care_tv_sqrtcd4_by_sex_2015
    store['mortality_threshold_by_sex_2015'] = mortality_threshold_by_sex_2015

    store['mortality_in_care_by_sex_race_2015'] = mortality_in_care_by_sex_race_2015
    store['mortality_in_care_age_by_sex_race_2015'] = mortality_in_care_age_by_sex_race_2015
    store['mortality_in_care_sqrtcd4_by_sex_race_2015'] = mortality_in_care_sqrtcd4_by_sex_race_2015
    store['mortality_out_care_by_sex_race_2015'] = mortality_out_care_by_sex_race_2015
    store['mortality_out_care_age_by_sex_race_2015'] = mortality_out_care_age_by_sex_race_2015
    store['mortality_out_care_tv_sqrtcd4_by_sex_race_2015'] = mortality_out_care_tv_sqrtcd4_by_sex_race_2015
    store['mortality_threshold_by_sex_race_2015'] = mortality_threshold_by_sex_race_2015

    store['mortality_in_care_by_sex_race_risk_2015'] = mortality_in_care_by_sex_race_risk_2015
    store['mortality_in_care_age_by_sex_race_risk_2015'] = mortality_in_care_age_by_sex_race_risk_2015
    store['mortality_in_care_sqrtcd4_by_sex_race_risk_2015'] = mortality_in_care_sqrtcd4_by_sex_race_risk_2015
    store['mortality_out_care_by_sex_race_risk_2015'] = mortality_out_care_by_sex_race_risk_2015
    store['mortality_out_care_age_by_sex_race_risk_2015'] = mortality_out_care_age_by_sex_race_risk_2015
    store['mortality_out_care_tv_sqrtcd4_by_sex_race_risk_2015'] = mortality_out_care_tv_sqrtcd4_by_sex_race_risk_2015
    store['mortality_threshold_by_sex_race_risk_2015'] = mortality_threshold_by_sex_race_risk_2015

    store['mortality_threshold_idu_5x'] = mortality_threshold_idu_5x
    store['mortality_threshold_idu_10x'] = mortality_threshold_idu_10x
