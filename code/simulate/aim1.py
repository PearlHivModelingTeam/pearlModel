# Imports
import sys
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
with pd.HDFStore(proc_dir + '/converted.h5') as store:
    on_art_2009 = store['on_art_2009'] 
    #naaccord = store['naaccord']
    #naaccord_2009 = store['naaccord_2009']
    mixture_2009_coeff = store['mixture_2009_coeff']
    naaccord_prop_2009 = store['naaccord_prop_2009']
    init_sqrtcd4n_coeff = store['init_sqrtcd4n_coeff']
    new_dx = store['new_dx']
    new_dx_interval = store['new_dx_interval']
    #init_age_gmix_coeffs = store['init_age_gmix_coeffs']
    mixture_h1yy_coeff= store['mixture_h1yy_coeff']
    #coeff_age_2009_ci = store['coeff_age_2009_ci']
    #coeff_cd4_decrease = store['coeff_cd4_decrease']
    #coeff_cd4_increase = store['coeff_cd4_increase']
    #coeff_mortality_out = store['coeff_mortality_out']
    #coeff_mortality_in = store['coeff_mortality_in']
    #coeff_ltfu = store['coeff_ltfu']
    #pctls_ltfu = store['pctls_ltfu']

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

def make_pop_2009(on_art_2009, mixture_2009_coeff, naaccord_prop_2009, init_sqrtcd4n_coeff, group_name):
    """ Create initial 2009 population. Draw ages from a mixed normal distribution truncated at 18 and 85. h1yy is assigned 
    using proportions from naaccord data. Finally, sqrt cd4n is drawn from a 0-truncated normal for each h1yy"""
    
    # Get coefficients 
    mus = [mixture_2009_coeff['mu1'], mixture_2009_coeff['mu2']]
    sigmas = [mixture_2009_coeff['sigma1'], mixture_2009_coeff['sigma2']]
    lambda1 = mixture_2009_coeff['lambda1']
    lambdas = [lambda1, 1.0 - lambda1] 

    # Define size of group population
    pop_size = on_art_2009[0].astype('int')

    # Draw population size for each component of the mixed normal
    components = np.random.choice([1,2], size=pop_size, p=lambdas, replace=True)
    pop_size_1 = (components == 1).sum()
    pop_size_2 = (components == 2).sum()

    # Draw age from each respective truncated normal
    if(pop_size_1 == 0):
        population = draw_from_trunc_norm(18, 85, mus[1], sigmas[1], pop_size_2)
    else:
        pop1 = draw_from_trunc_norm(18, 85, mus[0], sigmas[0], pop_size_1)
        pop2 = draw_from_trunc_norm(18, 85, mus[1], sigmas[1], pop_size_2)
        population = np.concatenate((pop1, pop2))
    
    # Create DataFrame
    population = pd.DataFrame(data={'age': population})

    # Create age categories
    population.age = np.floor(population.age)
    population['age_cat'] = np.floor(population.age / 10)
    population.loc[population['age_cat'] > 7, 'age_cat'] = 7
    population['id'] = range(population.index.size)
    population = population.sort_values('age')
    population = population.set_index(['age_cat', 'id'])

    # Assign H1YY to match naaccord distribution from naaccord_prop_2009
    for age_cat, grouped in population.groupby('age_cat'):
        if (naaccord_prop_2009.index.isin([(group_name, age_cat, 2000.0)]).any() ): 
            h1yy_data = naaccord_prop_2009.loc[(group_name, age_cat)]
        else: # replace missing data
            h1yy_data = naaccord_prop_2009.loc[('msm_white_male', age_cat)]
        population.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data.index.values, size=grouped.shape[0], p=h1yy_data.pct.values)

    # Pull cd4 count coefficients
    mean_intercept = init_sqrtcd4n_coeff['meanint']
    mean_slope = init_sqrtcd4n_coeff['meanslp']
    std_intercept = init_sqrtcd4n_coeff['stdint']
    std_slope = init_sqrtcd4n_coeff['stdslp']

    # Reindex for group operation
    population['h1yy'] = population['h1yy'].astype(int)
    population = population.reset_index().set_index(['h1yy', 'id']).sort_index()

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    for h1yy, group in population.groupby(level=0):
        mu = mean_intercept + (h1yy * mean_slope)
        sigma = std_intercept + (h1yy * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.inf, mu, sigma, size)
        population.loc[(h1yy,),'sqrt_init_cd4n'] = sqrt_cd4n

    return(population.reset_index().set_index('id').sort_index())

def simulate_new_dx(new_dx, dx_interval):
    """ Draw number of new diagnoses from a uniform distribution between upper and lower bounds. Calculate number of new art initiators by 
    assuming 75% link in the first year, then another 10% over the next three years. Assume 75% of these initiate art """
    
    # Draw new dx from a uniform distribution between upper and lower for 2016-2030 
    dx_interval = dx_interval[dx_interval.index > 2015].copy()
    dx_interval['rand'] = np.random.uniform(size=len(dx_interval.index)) 
    dx_interval['n_dx'] = dx_interval['lower'] + (dx_interval['upper'] - dx_interval['lower']) * dx_interval['rand']
    new_dx = pd.concat([new_dx, dx_interval.filter(items=['n_dx'])])
    new_dx = np.floor(new_dx)
    
    # We assume 75% link to care in the first year and a further 10% link in the next 3 years with equal probability
    new_dx['lag_step'] = new_dx['n_dx'] * 0.1 * (1./3.)
    new_dx['year0'] = new_dx['n_dx'] * 0.75
    new_dx['year1'] = new_dx['lag_step'].shift(1, fill_value=0)
    new_dx['year2'] = new_dx['lag_step'].shift(2, fill_value=0)
    new_dx['year3'] = new_dx['lag_step'].shift(3, fill_value=0)
    new_dx['total_linked'] = new_dx['year0'] + new_dx['year1'] + new_dx['year2'] + new_dx['year3']

    new_dx['n_art_init'] = (new_dx['total_linked'] * 0.75).astype(int)

    return(new_dx.filter(items=['n_art_init']))

def simulate_age(art_init_sim, mixture_h1yy_coeff):
    """ Draw ages for new art initiators """ 

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

    # Replace negative values with 0
    mixture_h1yy_coeff[mixture_h1yy_coeff < 0] = 0
   
    # Split into before and after 2018
    sim_coeff = mixture_h1yy_coeff.loc[mixture_h1yy_coeff.index.get_level_values('h1yy') >= 2018].copy()
    observed_coeff = mixture_h1yy_coeff.loc[mixture_h1yy_coeff.index.get_level_values('h1yy') < 2018].copy().rename(columns = {'pred': 'sim'})
    observed_coeff = pd.pivot_table(observed_coeff.reset_index(), values='sim', index='h1yy', columns='param')
    
    # Pull coefficients in 2018 
    sim_coeff['pred18'] = np.nan
    for name, group in sim_coeff.groupby('param'):
        sim_coeff.loc[(name, ), 'pred18'] = sim_coeff.loc[(name, 2018), 'pred'] 

    # Draw uniformly between predicted coeffs and coeff in 2018
    sim_coeff['rand'] = np.random.rand(len(sim_coeff.index))
    sim_coeff['min'] = np.minimum(sim_coeff['pred'], sim_coeff['pred18'])
    sim_coeff['max'] = np.maximum(sim_coeff['pred'], sim_coeff['pred18'])
    sim_coeff['sim'] = sim_coeff['min'] + sim_coeff['rand'] * (sim_coeff['max'] - sim_coeff['min'])

    # Reorganize table and glue them back together
    sim_coeff = pd.pivot_table(sim_coeff.reset_index(), values='sim', index='h1yy', columns='param')
    sim_coeff = pd.concat([observed_coeff, sim_coeff])

    # Lambdas should add to 1.0
    sim_coeff['lambda2'] = 1.0 - sim_coeff['lambda1']

    print(art_init_sim)
    print(sim_coeff)
    

###############################################################################
# Simulate Function                                                           #
###############################################################################

def simulate(group_name):
    """ Run one replication of the pearl model for a given group"""
    
    # Create 2009 population
    population_2009 = make_pop_2009(on_art_2009.loc[group_name], mixture_2009_coeff.loc[group_name], naaccord_prop_2009.copy(), init_sqrtcd4n_coeff.loc[group_name], group_name)

    # Simulate number of new art initiators
    art_init_sim = simulate_new_dx(new_dx.loc[group_name].copy(), new_dx_interval.loc[group_name].copy())

    # Simulate the ages of new art initiators
    new_art_age_mixed = simulate_age(art_init_sim.copy(), mixture_h1yy_coeff.loc[group_name].copy())


###############################################################################
# Main Function                                                               #
###############################################################################

def main():
    for group_name in on_art_2009.index.values:
        print(group_name)
        simulate(group_name)
        

def main1():
    group_name = 'het_black_female'
    print(group_name)
    simulate(group_name)

main1()
