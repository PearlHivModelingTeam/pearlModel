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
proc_dir = cwd + '/../../data/processed'
out_dir = cwd + '/../out'

# Load everything
with pd.HDFStore(proc_dir + '/preprocessed.h5') as store:
    on_art = store['on_art']
    naaccord = store['naaccord']
    naaccord_2009 = store['naaccord_2009']
    naaccord_prop_2009 = store['naaccord_prop_2009']
    init_sqrtcd4n_coeffs = store['init_sqrtcd4n_coeffs']
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

def make_pop_2009(on_art, coeff_age_2009_ci, naaccord_prop_2009, init_sqrtcd4n_coeffs, group_name):
    """ Create initial 2009 population. Draw ages from a mixed normal distribution truncated at 18 and 85. Mixed normal coefficient are
    drawn uniformly from the 95% CI. h1yy is assigned using proportions from naaccord data. Finally, sqrt cd4n is drawn from a 0-truncated
    normal for each h1yy"""

    def draw_from_trunc_norm(a, b, mu, sigma, size):
        """ Draws `size` values from a truncated normal with the given parameters """
        a_mod = (a - mu) / sigma
        b_mod = (b - mu) / sigma
        return stats.truncnorm.rvs(a_mod, b_mod, loc=mu, scale=sigma, size=size)

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
    population = draw_from_trunc_norm(18, 85, mus[0], sigmas[0], pop_size_1)
    if(group_name != 'idu_hisp_female'):
        pop2 = draw_from_trunc_norm(18, 85, mus[1], sigmas[1], pop_size_2)
        population = np.concatenate((population, pop2))
    
    # Create DataFrame
    population = pd.DataFrame(data={'age': population})

    # Create age categories
    population.age = np.floor(population.age)
    population['age_cat'] = np.floor(population.age / 10)
    population.loc[population.age_cat > 7, 'age_cat'] = 7
    population['id'] = range(population.index.size)
    population = population.sort_values('age')
    population = population.set_index(['age_cat', 'id'])

    # Assign H1YY to match naaccord distribution from naaccord_prop_2009
    for age_cat, grouped in population.groupby('age_cat'):
        h1yy_data = naaccord_prop_2009.loc[(group_name, age_cat)]
        population.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data.index.values, size=grouped.shape[0], p=h1yy_data.pct.values)

    # Pull cd4 count coefficients
    mean_intercept = init_sqrtcd4n_coeffs.loc[group_name, 'meanint']
    mean_slope = init_sqrtcd4n_coeffs.loc[group_name, 'meanslope']
    std_intercept = init_sqrtcd4n_coeffs.loc[group_name, 'stdint']
    std_slope = init_sqrtcd4n_coeffs.loc[group_name, 'stdslope']

    # Reindex for group operation
    population.h1yy = population.h1yy.astype(int)
    population = population.reset_index().set_index(['h1yy', 'id']).sort_index()

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    for h1yy, group in population.groupby(level=0):
        mu = mean_intercept + (h1yy * mean_slope)
        sigma = std_intercept + (h1yy * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.inf, mu, sigma, size)
        population.loc[(h1yy,),'sqrt_init_cd4n'] = sqrt_cd4n

    return(population.reset_index().set_index('id').sort_index())


###############################################################################
# Simulate Function                                                           #
###############################################################################

def simulate(group_name):
    """ Run one replication of the pearl model for a given group"""
    
    # Filter population to one group
    naaccord_group = filter_group(naaccord, group_name)
    naaccord_2009_group = filter_group(naaccord_2009, group_name)
    
    # Create 2009 population
    population = make_pop_2009(on_art, coeff_age_2009_ci, naaccord_prop_2009, init_sqrtcd4n_coeffs, group_name)

###############################################################################
# Main Function                                                               #
###############################################################################

def main():
    simulate('idu_hisp_female')

main()
