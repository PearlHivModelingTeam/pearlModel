# Imports
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# Stat modeling packages
import scipy.stats as stats
#from statsmodels.gam.api import GLMGam, BSplines
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import sklearn.mixture as skm

# R to python interface
#import rpy2.robjects as ro
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.packages import importr

# Activate R interface
#base = importr('base')
#pandas2ri.activate()

# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../data/processed'
out_dir = cwd + '/../data/out'

# Load everything
with pd.HDFStore(in_dir + '/preprocessed.h5') as store:
    on_art = store['on_art']
    naaccord = store['naaccord']
    naaccord_2009 = store['naaccord_2009']
    naaccord_prop_2009 = store['naaccord_prop_2009']
    init_cd4n_coeffs = store['init_cd4n_coeffs']
    new_dx = store['new_dx']
    dx_interval = store['dx_interval']
    init_age_gmix_coeffs = store['init_age_gmix_coeffs']
    gmix_param_coeffs = store['gmix_param_coeffs']
    coeff_age_2009_ci = store['coeff_age_2009_ci']
    coeff_cd4_decrease = store['coeff_cd4_decrease']
    coeff_cd4_increase = store['coeff_cd4_increase']
    coeff_mortality_out = store['coeff_mortality_out']
    coeff_mortality_in = store['coeff_mortality_in']
    coeff_ltfu = store['coeff_ltfu']
    pctls_ltfu = store['pctls_ltfu']

###############################################################################
# Functions                                                                   #
###############################################################################

def filter_group(df, group_name):
    """Filter to a single group"""
    df = df.loc[df['group'] == group_name]
    return df

def make_pop_2009(on_art, coeff_age_2009_ci, naaccord_prop_2009, init_cd4n_coeffs, group_name):
    """ Not sure yet """

    # Sample from 95% confidence intervals
    mus = [np.random.uniform(coeff_age_2009_ci.loc[group_name, 'mu1_p025'], coeff_age_2009_ci.loc[group_name, 'mu1_p975']), 
           np.random.uniform(coeff_age_2009_ci.loc[group_name, 'mu2_p025'], coeff_age_2009_ci.loc[group_name, 'mu2_p975'])]
    sigmas = [np.random.uniform(coeff_age_2009_ci.loc[group_name, 'sigma1_p025'], coeff_age_2009_ci.loc[group_name, 'sigma1_p975']),
              np.random.uniform(coeff_age_2009_ci.loc[group_name, 'sigma2_p025'], coeff_age_2009_ci.loc[group_name, 'sigma2_p975'])]
    lambda1 = np.random.uniform(coeff_age_2009_ci.loc[group_name, 'lambda1_p025'], coeff_age_2009_ci.loc[group_name, 'lambda1_p975'])
    lambdas = [lambda1, 1.0 - lambda1]

    # Define size of group population
    pop_size = on_art.loc[group_name][0].astype('int')

    # Draw population size for each component of the mixed normal
    components = np.random.choice([1,2], size=pop_size, p=lambdas, replace=True)
    pop_size_1 = (components == 1).sum()
    pop_size_2 = (components == 2).sum()

    # Draw age from each respective truncated normal
    pop1 = stats.truncnorm.rvs((18 - mus[0]) / sigmas[0], (85 - mus[0]) / sigmas[0], loc=mus[0], scale=sigmas[0], size=pop_size_1)
    pop2 = stats.truncnorm.rvs((18 - mus[1]) / sigmas[1], (85 - mus[1]) / sigmas[1], loc=mus[1], scale=sigmas[1], size=pop_size_2)
    
    # Create DataFrame
    mixed_pop = np.concatenate((pop1, pop2))
    mixed_pop = pd.DataFrame(data={'age': mixed_pop})

    # Create age categories
    mixed_pop.age = np.floor(mixed_pop.age)
    mixed_pop['age_cat'] = np.floor(mixed_pop.age / 10)
    mixed_pop.loc[mixed_pop.age_cat > 7, 'age_cat'] = 7

    mixed_pop['id'] = range(mixed_pop.index.size)
    mixed_pop = mixed_pop.sort_values('age')
    mixed_pop = mixed_pop.set_index(['age_cat', 'id'])
 
    # Assign H1YY to match naaccord distribution from naaccord_prop_2009
    for age_cat, grouped in mixed_pop.groupby('age_cat'):
        h1yy_data = naaccord_prop_2009.loc[(group_name, age_cat)]
        mixed_pop.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data.index.values, size=grouped.shape[0], p=h1yy_data.pct.values)

    # Pull cd4 count coefficients
    print(init_cd4n_coeffs)
    mean_intercept = init_cd4n_coeffs.loc[group_name, 'meanint']
    mean_slope = init_cd4n_coeffs.loc[group_name, 'meanslope']
    std_intercept = init_cd4n_coeffs.loc[group_name, 'stdint']
    std_slope = init_cd4n_coeffs.loc[group_name, 'stdslope']
    print(mean_intercept)
    print(mean_slope)
    print(std_intercept)
    print(std_slope)






    #print(mixed_pop)


###############################################################################
# Simulate Function                                                           #
###############################################################################

def simulate(group_name):
    """ Run one replication of the pearl model for a given group"""
    
    # Filter population to one group
    naaccord_group = filter_group(naaccord, group_name)
    naaccord_2009_group = filter_group(naaccord_2009, group_name)
    
    # Create 2009 population
    make_pop_2009(on_art, coeff_age_2009_ci, naaccord_prop_2009, init_cd4n_coeffs, group_name)

###############################################################################
# Main Function                                                               #
###############################################################################

def main():
    simulate('het_black_female')

main()
