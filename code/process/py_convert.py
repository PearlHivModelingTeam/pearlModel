# Imports
import os
import numpy as np
import pandas as pd

# R to python interface
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate R interface
base = importr('base')
pandas2ri.activate()

# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../../data/input'
param_dir = cwd + '/../../data/param'
proc_dir = cwd + '/../../data/processed'

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


robjects.r.source('r_convert.r')

# Number of people on art in 2009: on_art_2009
on_art_2009 = (robjects.r['on_art']).set_index('group')

# Proportion of people with certain h1yy given age, risk, sex: naaccord_prop_2009
naaccord_prop_2009 = robjects.r['naaccord_prop_2009']
naaccord_prop_2009.columns = map(str.lower, naaccord_prop_2009.columns)
naaccord_prop_2009 = naaccord_prop_2009.set_index(['group', 'age2009cat', 'h1yy'])

# Mean and std of sqrtcd4n as a glm of h1yy for each group in 2009: init_sqrtcd4n_coeff_2009
init_sqrtcd4n_coeff_2009 = (robjects.r['init_sqrtcd4n_coeff_2009']).set_index('group')

# Mixed gaussian coefficients for age of patients alive in 2009: mixture_2009
mixture_2009_coeff = (robjects.r['mixture_2009_coeff']).set_index('group')

# New dx and dx prediction intervals
new_dx = (robjects.r['new_dx']).set_index(['group', 'year'])
new_dx_interval = (robjects.r['new_dx_interval']).set_index(['group', 'year'])

# Age at haart init mixed gaussian coefficients
mixture_h1yy_coeff = robjects.r['mixture_h1yy_coeff']
mixture_h1yy_coeff.columns = map(str.lower, mixture_h1yy_coeff.columns) 
mixture_h1yy_coeff = mixture_h1yy_coeff.set_index(['group', 'param', 'h1yy'])

# Mean and std of sqrtcd4n as a glm of h1yy for each group in 2009: init_sqrtcd4n_coeff_2009
init_sqrtcd4n_coeff = (robjects.r['init_sqrtcd4n_coeff']).set_index('group')

# Coefficients of cd4 increase over time
cd4_increase_coeff = clean_coeff(pd.read_sas(param_dir + '/cd4_increase_coeff_190508.sas7bdat'))
cd4_increase_coeff['p5'] = 1.0 # Define percentiles for time_from_h1yy
cd4_increase_coeff['p35'] = 3.0
cd4_increase_coeff['p65'] = 6.0
cd4_increase_coeff['p95'] = 12.0

# Coefficients for cd4 decline out of care
cd4_decrease_coeff = pd.read_sas(param_dir + '/coeff_cd4_decrease_190508.sas7bdat')
cd4_decrease_coeff.columns = map(str.lower, cd4_decrease_coeff.columns)

# Coefficients for loss to follow up
ltfu_coeff = clean_coeff(pd.read_sas(param_dir + '/coeff_ltfu_190508.sas7bdat'))
ltfu_pctls = clean_coeff(pd.read_sas(param_dir + '/pctls_ltfu_190508.sas7bdat'))
ltfu_coeff = pd.concat([ltfu_coeff, ltfu_pctls], axis=1)

# Coefficients for mortality in care
mortality_in_care_coeff = (robjects.r['mortality_in_care_coeff']).set_index('group')

# Coefficients for mortality out of care
mortality_out_care_coeff = pd.read_sas(param_dir + '/coeff_mortality_out_care_190508.sas7bdat')
mortality_out_care_coeff.columns = map(str.lower, mortality_out_care_coeff.columns)
mortality_out_care_coeff = clean_coeff(mortality_out_care_coeff)

# Probability to reengage in care for each group
prob_reengage = clean_coeff(pd.read_csv(param_dir + '/prob_reengage.csv'))

# Save everything
with pd.HDFStore(proc_dir + '/converted.h5') as store:
    store['on_art_2009'] = on_art_2009
    store['naaccord_prop_2009'] = naaccord_prop_2009 
    store['init_sqrtcd4n_coeff_2009'] = init_sqrtcd4n_coeff_2009
    store['mixture_2009_coeff'] = mixture_2009_coeff
    store['new_dx'] = new_dx
    store['new_dx_interval'] = new_dx_interval
    store['mixture_h1yy_coeff'] = mixture_h1yy_coeff
    store['init_sqrtcd4n_coeff'] = init_sqrtcd4n_coeff
    store['cd4_increase_coeff'] = cd4_increase_coeff
    store['ltfu_coeff'] = ltfu_coeff
    store['mortality_in_care_coeff'] = mortality_in_care_coeff
    store['mortality_out_care_coeff'] = mortality_out_care_coeff
    store['cd4_decrease_coeff'] = cd4_decrease_coeff
    store['prob_reengage'] = prob_reengage

