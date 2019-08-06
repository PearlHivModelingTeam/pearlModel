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

def draw_from_trunc_norm(a, b, mu, sigma, size):
    """ Draws `size` values from a truncated normal with the given parameters """
    a_mod = (a - mu) / sigma
    b_mod = (b - mu) / sigma
    return stats.truncnorm.rvs(a_mod, b_mod, loc=mu, scale=sigma, size=size)

def make_pop_2009(on_art, coeff_age_2009_ci, naaccord_prop_2009, init_sqrtcd4n_coeffs, group_name):
    """ Create initial 2009 population. Draw ages from a mixed normal distribution truncated at 18 and 85. Mixed normal coefficient are
    drawn uniformly from the 95% CI. h1yy is assigned using proportions from naaccord data. Finally, sqrt cd4n is drawn from a 0-truncated
    normal for each h1yy"""

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

def simulate_new_dx(new_dx, dx_interval, group_name):
    """ Draw number of new diagnoses from a uniform distribution between upper and lower bounds. Calculate number of new art initiators by 
    assuming 75% link in the first year, then another 10% over the next three years. Assume 75% of these initiate art """
    
    # Draw new dx from a uniform distribution between upper and lower for 2016-2030 
    new_dx = new_dx.loc[group_name]
    dx_interval = dx_interval.loc[group_name].iloc[7:]
    dx_interval['rand'] = np.random.uniform(size=len(dx_interval.index)) 
    dx_interval['n_dx'] = dx_interval.lower + (dx_interval.upper - dx_interval.lower) * dx_interval.rand
    new_dx = pd.concat([new_dx, dx_interval.filter(items=['n_dx'])])
    new_dx = np.floor(new_dx)
  
    # We assume 75% link to care in the first year and a further 10% link in the next 3 years with equal probability
    new_dx['lag_step'] = new_dx.n_dx * 0.1 * (1./3.)
    new_dx['year0'] = new_dx.n_dx * 0.75
    new_dx['year1'] = new_dx.lag_step.shift(1, fill_value=0)
    new_dx['year2'] = new_dx.lag_step.shift(2, fill_value=0)
    new_dx['year3'] = new_dx.lag_step.shift(3, fill_value=0)
    new_dx['total_linked'] = new_dx['year0'] + new_dx['year1'] + new_dx['year2'] + new_dx['year3']

    # TODO why take off another 25%?
    new_dx['n_art_init'] = (new_dx.total_linked * 0.75).astype(int)

    return(new_dx.filter(items=['n_art_init']))

def simulate_age(art_init_sim, gmix_param_coeffs, group_name):
    """ Draw ages for new art initiators """ 
    gmix_param_coeffs = gmix_param_coeffs.loc[group_name]

    def sim_pop(coeffs, pop_size):
        """ Pick from young normal, old normal, or mixed normal and draw ages """
        print(coeffs)
        print(pop_size)
        
        if (coeffs.model == 'mix'):
            components = np.random.choice([1,2], size=pop_size, p=[coeffs.weight1, coeffs.weight2], replace=True)
            pop_size_1 = (components == 1).sum()
            pop_size_2 = (components == 2).sum()

            #pop1 = draw_from_trunc_norm(18, 85, coeffs.mu1, , pop_size_1)
        elif (coeffs.model == 'young'):
            pass
        else:
            pass


    print(gmix_param_coeffs.to_string())
    print(art_init_sim)

    sim_pop(gmix_param_coeffs.loc[2009], art_init_sim.loc[2009].iloc[0]) 

###############################################################################
# Simulate Function                                                           #
###############################################################################

def simulate(group_name):
    """ Run one replication of the pearl model for a given group"""
    
    # Filter population to one group
    naaccord_group = filter_group(naaccord, group_name)
    naaccord_2009_group = filter_group(naaccord_2009, group_name)
    
    # Create 2009 population
    population_2009 = make_pop_2009(on_art, coeff_age_2009_ci, naaccord_prop_2009.copy(), init_sqrtcd4n_coeffs, group_name)

    # Simulate number of new art initiators
    art_init_sim = simulate_new_dx(new_dx.copy(), dx_interval.copy(), group_name)

    # Simulate the ages of new art initiators
    new_art_age_mixed = simulate_age(art_init_sim.copy(), gmix_param_coeffs, group_name)


###############################################################################
# Main Function                                                               #
###############################################################################

def main():
    simulate('het_black_female')

main()
