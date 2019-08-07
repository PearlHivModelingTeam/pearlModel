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

robjects.r.source('r_convert.r')

# Number of people on art in 2009: on_art_2009
on_art_2009 = (robjects.r['on_art']).set_index('group')

# Proportion of people with certain h1yy given age, risk, sex: naaccord_prop_2009
naaccord_prop_2009 = robjects.r['naaccord_prop_2009']
naaccord_prop_2009.columns = map(str.lower, naaccord_prop_2009.columns)
naaccord_prop_2009 = naaccord_prop_2009.set_index(['group', 'age2009cat', 'h1yy'])

# Mean and std of sqrtcd4n as a glm of h1yy for each group in 2009: init_sqrtcd4n_coeffs 
init_sqrtcd4n_coeffs = (robjects.r['init_sqrtcd4n_coeffs']).set_index('group')

# Mixed gaussian coefficients for age of patients alive in 2009: mixture_2009
mixture_2009 = (robjects.r['mixture_2009']).set_index('group')

# New dx and dx prediction intervals
new_dx = (robjects.r['new_dx']).set_index(['group', 'year'])
new_dx_interval = (robjects.r['new_dx_interval']).set_index(['group', 'year'])

with pd.HDFStore(proc_dir + '/converted.h5') as store:
    store['on_art_2009'] = on_art_2009
    store['naaccord_prop_2009'] = naaccord_prop_2009 
    store['init_sqrtcd4n_coeffs'] = init_sqrtcd4n_coeffs
    store['mixture_2009'] = mixture_2009

    store['new_dx'] = new_dx
    store['new_dx_interval'] = new_dx_interval
