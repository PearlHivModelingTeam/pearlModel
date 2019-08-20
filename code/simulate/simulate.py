# Imports
import sys
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import namedtuple

# print more rows
pd.options.display.max_rows = 150

# Define directories
cwd = os.getcwd()
proc_dir = cwd + '/../../data/processed'
out_dir = cwd + '/../../out'

# Load everything
with pd.HDFStore(proc_dir + '/converted.h5') as store:
    on_art_2009 = store['on_art_2009'] 
    mixture_2009_coeff = store['mixture_2009_coeff']
    naaccord_prop_2009 = store['naaccord_prop_2009']
    init_sqrtcd4n_coeff_2009 = store['init_sqrtcd4n_coeff_2009']
    new_dx = store['new_dx']
    new_dx_interval = store['new_dx_interval']
    mixture_h1yy_coeff= store['mixture_h1yy_coeff']
    init_sqrtcd4n_coeff = store['init_sqrtcd4n_coeff']
    cd4_increase_coeff = store['cd4_increase_coeff']
    cd4_decrease_coeff = store['cd4_decrease_coeff']
    ltfu_coeff = store['ltfu_coeff']
    mortality_in_care_coeff = store['mortality_in_care_coeff']
    mortality_out_care_coeff = store['mortality_out_care_coeff']
    prob_reengage = store['prob_reengage']

###############################################################################
# Functions                                                                   #
###############################################################################

def filter_group(df, group_name):
    """Filter to a single group"""
    df = df.loc[df['group'] == group_name]
    return df

def draw_from_trunc_norm(a, b, mu, sigma, n):
    """ Draws n values from a truncated normal with the given parameters """
    a_mod = (a - mu) / sigma
    b_mod = (b - mu) / sigma
    return stats.truncnorm.rvs(a_mod, b_mod, loc=mu, scale=sigma, size=n)
    
def sim_pop(coeffs, pop_size):
    """ Draw ages from a mixed or single gaussian truncated at 18 and 85 given the coefficients and population size."""
    
    components = np.random.choice([1,2], size=pop_size, p=[coeffs['lambda1'], coeffs['lambda2']], replace=True)
    pop_size_1 = (components == 1).sum()
    pop_size_2 = (components == 2).sum()

    # Draw age from each respective truncated normal
    if(pop_size_1 == 0):
        population = draw_from_trunc_norm(18, 85, coeffs['mu2'], coeffs['sigma2'], pop_size_2)
    elif(pop_size_2 == 0):
        population = draw_from_trunc_norm(18, 85, coeffs['mu1'], coeffs['sigma1'], pop_size_1)
    else:
        pop1 = draw_from_trunc_norm(18, 85, coeffs['mu1'], coeffs['sigma1'], pop_size_1)
        pop2 = draw_from_trunc_norm(18, 85, coeffs['mu2'], coeffs['sigma2'], pop_size_2)
        population = np.concatenate((pop1, pop2))

    # Create DataFrame
    population = pd.DataFrame(data={'age': population})

    return population

def initialize_cd4n_cat(pop):
    """ Given inital sqrtcd4n,  add columns with categories used for cd4 increase function """
    pop['init_cd4_cat'] = np.select([pop['init_sqrtcd4n'].lt(np.sqrt(200.0)),
                                            pop['init_sqrtcd4n'].ge(np.sqrt(200.0)) & pop['init_sqrtcd4n'].lt(np.sqrt(350.0)),
                                            pop['init_sqrtcd4n'].ge(np.sqrt(350.0)) & pop['init_sqrtcd4n'].lt(np.sqrt(500.0)),
                                            pop['init_sqrtcd4n'].ge(np.sqrt(500.0))], 
                                            [1, 2, 3, 4])
    pop['cd4_cat_349'] = (pop['init_cd4_cat'] == 2).astype(int)
    pop['cd4_cat_499'] = (pop['init_cd4_cat'] == 3).astype(int)
    pop['cd4_cat_500'] = (pop['init_cd4_cat'] == 4).astype(int)

    pop = pop.drop(['init_cd4_cat'], axis=1)

    return pop

def calculate_time_varying_cd4n(pop, coeffs, year):
    """ Calculate time varying cd4n as a linear function of age_cat, cd4_cat, time_from_h1yy and their cross terms """
    
    # Calculate spline variables
    pop['time_from_h1yy']    =  year - pop['h1yy']
    pop['time_from_h1yy_']   = (np.maximum(0, pop['time_from_h1yy'] - coeffs['p5' ])**2 -
                                np.maximum(0, pop['time_from_h1yy'] - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    pop['time_from_h1yy__']  = (np.maximum(0, pop['time_from_h1yy'] - coeffs['p35'])**2 -
                                np.maximum(0, pop['time_from_h1yy'] - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    pop['time_from_h1yy___'] = (np.maximum(0, pop['time_from_h1yy'] - coeffs['p65'])**2 -
                                np.maximum(0, pop['time_from_h1yy'] - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])

    # Calculate time varying sqrt cd4n
    pop['time_varying_sqrtcd4n'] = (coeffs['intercept_c'] +
                                   (coeffs['agecat_c'] * pop['age_cat']) + 
                                   (coeffs['cd4cat349_c'] * pop['cd4_cat_349']) +
                                   (coeffs['cd4cat499_c'] * pop['cd4_cat_499']) +
                                   (coeffs['cd4cat500_c'] * pop['cd4_cat_500']) +
                                   (coeffs['time_from_h1yy_c'] * pop['time_from_h1yy']) +
                                   (coeffs['_time_from_h1yy_c'] * pop['time_from_h1yy_']) +
                                   (coeffs['__time_from_h1yy_c'] * pop['time_from_h1yy__']) +
                                   (coeffs['___time_from_h1yy_c'] * pop['time_from_h1yy___']) +
                                   (coeffs['_time_from_cd4cat349_c'] * pop['time_from_h1yy_'] * pop['cd4_cat_349']) +
                                   (coeffs['_time_from_cd4cat499_c'] * pop['time_from_h1yy_'] * pop['cd4_cat_499']) +
                                   (coeffs['_time_from_cd4cat500_c'] * pop['time_from_h1yy_'] * pop['cd4_cat_500']) +
                                   (coeffs['__time_fro_cd4cat349_c'] * pop['time_from_h1yy__'] * pop['cd4_cat_349']) +
                                   (coeffs['__time_fro_cd4cat499_c'] * pop['time_from_h1yy__'] * pop['cd4_cat_499']) +
                                   (coeffs['__time_fro_cd4cat500_c'] * pop['time_from_h1yy__'] * pop['cd4_cat_500']) +
                                   (coeffs['___time_fr_cd4cat349_c'] * pop['time_from_h1yy___'] * pop['cd4_cat_349']) +
                                   (coeffs['___time_fr_cd4cat499_c'] * pop['time_from_h1yy___'] * pop['cd4_cat_499']) +
                                   (coeffs['___time_fr_cd4cat500_c'] * pop['time_from_h1yy___'] * pop['cd4_cat_500'])) 

    return pop['time_varying_sqrtcd4n'].values

def calculate_cd4n_decrease(pop, coeffs, year):
    """ Calculate new time varying cd4 count for population out of care """
    time_out = year - pop['ltfu_year']
    diff = (coeffs['time_out_of_naaccord_c'] * time_out) + (coeffs['sqrtcd4_exit_c'] * pop['sqrtcd4n_exit']) +  coeffs['intercept_c'] 
    time_varying_sqrtcd4n = np.sqrt((pop['sqrtcd4n_exit']**2) * np.exp(diff) * 1.5)
    
    return time_varying_sqrtcd4n.values

def calculate_ltfu_prob(pop, coeffs, year):
    """ Calculate the probability of loss to follow up """

    # Calculate spline variables
    pop['age_']   = (np.maximum(0, pop['age'] - coeffs['p5'])**2 - 
                     np.maximum(0, pop['age'] - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    pop['age__']  = (np.maximum(0, pop['age'] - coeffs['p35'])**2 - 
                     np.maximum(0, pop['age'] - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    pop['age___'] = (np.maximum(0, pop['age'] - coeffs['p65'])**2 - 
                     np.maximum(0, pop['age'] - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    
    # Create haart_period variable
    pop['haart_period'] = (pop['h1yy'] > 2010).astype(int)
    
    # Calculate log odds
    odds = (coeffs['intercept_c'] + 
           (coeffs['age_c'] * pop['age']) + 
           (coeffs['_age_c'] * pop['age_']) + 
           (coeffs['__age_c'] * pop['age__']) + 
           (coeffs['___age_c'] * pop['age___']) + 
           (coeffs['year_c'] * year ) +
           (coeffs['sqrtcd4n_c'] * pop['init_sqrtcd4n']) +
           (coeffs['haart_period_c'] * pop['haart_period']))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob.values

def calculate_death_in_care_prob(pop, coeffs, year):
    """ Calculate the individual probability of dying in care """
    odds = (coeffs['intercept_est'] + 
           (coeffs['ageby10_est'] * pop['age_cat']) +
           (coeffs['sqrtcd4n_est'] * pop['init_sqrtcd4n']) +
           (coeffs['year_est'] * year) +
           (coeffs['h1yy_est'] * pop['h1yy']))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob.values

def calculate_death_out_care_prob(pop, coeffs, year):
    """ Calculate the individual probability of dying in care """
    odds = (coeffs['intercept_c'] + 
           (coeffs['agecat_c'] * pop['age_cat']) +
           (coeffs['tv_sqrtcd4n_c'] * pop['time_varying_sqrtcd4n']) +
           (coeffs['year_c'] * year))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob.values 

def make_pop_2009(on_art_2009, mixture_2009_coeff, naaccord_prop_2009, init_sqrtcd4n_coeff_2009, cd4_increase_coeff, group_name):
    """ Create initial 2009 population. Draw ages from a mixed normal distribution truncated at 18 and 85. h1yy is assigned 
    using proportions from naaccord data. Finally, sqrt cd4n is drawn from a 0-truncated normal for each h1yy"""

    # Draw ages from the truncated mixed gaussian
    mixture_2009_coeff['lambda2'] = 1.0 - mixture_2009_coeff['lambda1']
    pop_size = on_art_2009[0].astype('int')
    population = sim_pop(mixture_2009_coeff, pop_size)

    # Create age categories
    population['age'] = np.floor(population['age'])
    population['age_cat'] = np.floor(population['age'] / 10)
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
    mean_intercept = init_sqrtcd4n_coeff_2009['meanint']
    mean_slope = init_sqrtcd4n_coeff_2009['meanslp']
    std_intercept = init_sqrtcd4n_coeff_2009['stdint']
    std_slope = init_sqrtcd4n_coeff_2009['stdslp']

    # Reindex for group operation
    population['h1yy'] = population['h1yy'].astype(int)
    population = population.reset_index().set_index(['h1yy', 'id']).sort_index()

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    for h1yy, group in population.groupby(level=0):
        mu = mean_intercept + (h1yy * mean_slope)
        sigma = std_intercept + (h1yy * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.inf, mu, sigma, size)
        population.loc[(h1yy,),'init_sqrtcd4n'] = sqrt_cd4n

    population = population.reset_index().set_index('id').sort_index()

    # Toss out age_cat < 2
    population.loc[population['age_cat'] < 2, 'age_cat'] = 2

    # Calculate time varying cd4 count
    population = initialize_cd4n_cat(population)
    population['time_varying_sqrtcd4n'] = calculate_time_varying_cd4n(population.copy(), cd4_increase_coeff, 2009.0)

    # Allow for updated h1yy and init_sqrtcd4n on reengagement
    population['n_lost'] = 0
    population['h1yy_orig'] = population['h1yy']
    population['init_sqrtcd4n_orig'] = population['init_sqrtcd4n']

    return population

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

    return new_dx.filter(items=['n_art_init'])

def make_new_population(art_init_sim, mixture_h1yy_coeff, init_sqrtcd4n_coeff, cd4_increase_coeff, pop_size_2009):
    """ Draw ages for new art initiators """ 

    # Replace negative values with 0
    mixture_h1yy_coeff[mixture_h1yy_coeff < 0] = 0
   
    # Split into before and after 2018
    sim_coeff = mixture_h1yy_coeff.loc[mixture_h1yy_coeff.index.get_level_values('h1yy') >= 2018].copy()
    observed_coeff = mixture_h1yy_coeff.loc[mixture_h1yy_coeff.index.get_level_values('h1yy') < 2018].copy().rename(columns = {'pred': 'sim'})
    observed_coeff = pd.pivot_table(observed_coeff.reset_index(), values='sim', index='h1yy', columns='param').rename_axis(None, axis=1)
    
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
    sim_coeff = pd.pivot_table(sim_coeff.reset_index(), values='sim', index='h1yy', columns='param').rename_axis(None, axis=1)
    sim_coeff = pd.concat([observed_coeff, sim_coeff])

    # Lambdas should add to 1.0
    sim_coeff['lambda2'] = 1.0 - sim_coeff['lambda1']
    sim_coeff.loc[sim_coeff['lambda2'] < 0, 'lambda1'] = 1.0
    sim_coeff.loc[sim_coeff['lambda2'] < 0, 'lambda2'] = 0

    # Dont simulate new dxs in 2009
    sim_coeff = sim_coeff.drop(2009)

    new_population = pd.DataFrame()
    for h1yy, coeffs in sim_coeff.groupby('h1yy'):
        population = sim_pop(coeffs.iloc[0], art_init_sim.loc[h1yy, 'n_art_init'])
        population['h1yy'] = h1yy
        new_population = pd.concat([new_population, population])

    new_population['age'] = np.floor(new_population['age'])
    new_population['age_cat'] = np.floor(new_population['age'] / 10)
    new_population.loc[new_population['age_cat'] < 2, 'age_cat'] = 2
    new_population.loc[new_population['age_cat'] > 7, 'age_cat'] = 7
    
    # Add id number
    new_population['id'] = np.arange(pop_size_2009, (pop_size_2009 + new_population.index.size))

    # Pull cd4 count coefficients
    mean_intercept = init_sqrtcd4n_coeff['meanint']
    mean_slope = init_sqrtcd4n_coeff['meanslp']
    std_intercept = init_sqrtcd4n_coeff['stdint']
    std_slope = init_sqrtcd4n_coeff['stdslp']

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    new_population = new_population.set_index('h1yy')
    for h1yy, group in new_population.groupby(level=0):
        mu = mean_intercept + (h1yy * mean_slope)
        sigma = std_intercept + (h1yy * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.inf, mu, sigma, size)
        new_population.loc[h1yy, 'init_sqrtcd4n'] = sqrt_cd4n
   
    new_population = new_population.reset_index().set_index('id').sort_index()
    
    # Calculate time varying cd4 count
    new_population = initialize_cd4n_cat(new_population)
    new_population['time_varying_sqrtcd4n'] = new_population['init_sqrtcd4n']

    # Allow for updated h1yy and sqrtcd4n on reengagement
    new_population['n_lost'] = 0
    new_population['h1yy_orig'] = new_population['h1yy']
    new_population['init_sqrtcd4n_orig'] = new_population['init_sqrtcd4n']
    
    return new_population

class Pearl:
    
    def __init__(self, init_pop, new_pop, group_name, replication):
        self.group_name = group_name
        self.replication = replication
        self.year = 2009
        self.in_care = init_pop.copy()[new_pop.columns] # Use consistent column order
        self.new_in_care = pd.DataFrame()
        self.new_dx = new_pop.copy()
        self.out_care = pd.DataFrame()
        self.new_out_care = pd.DataFrame()
        self.dead_in_care = pd.DataFrame()
        self.dead_out_care = pd.DataFrame()
    
        # Allow loss to follow up to occur in initial year
        self.lose_to_follow_up()
        self.out_care = self.out_care.append(self.new_out_care.copy())

        #snapshots
        self.in_care_snap = pd.DataFrame()
        self.out_care_snap = pd.DataFrame()
        self.new_in_care_snap = pd.DataFrame()
        self.new_out_care_snap = pd.DataFrame()
        self.take_snapshot()

        # Move to 2010
        self.year += 1

    def increment_age(self):
        self.in_care['age'] = self.in_care['age'] + 1.0
        self.in_care['age_cat'] = np.floor(self.in_care['age'] / 10)
        self.in_care.loc[self.in_care['age_cat'] < 2, 'age_cat'] = 2
        self.in_care.loc[self.in_care['age_cat'] > 7, 'age_cat'] = 7
        
        self.out_care['age'] = self.out_care['age'] + 1.0
        self.out_care['age_cat'] = np.floor(self.out_care['age'] / 10)
        self.out_care.loc[self.out_care['age_cat'] < 2, 'age_cat'] = 2
        self.out_care.loc[self.out_care['age_cat'] > 7, 'age_cat'] = 7

    def increase_cd4_count(self):
        self.in_care['time_varying_sqrtcd4n'] = calculate_time_varying_cd4n(self.in_care.copy(), cd4_increase_coeff.loc[self.group_name], self.year)

    def decrease_cd4_count(self):
        self.out_care['time_varying_sqrtcd4n'] = calculate_cd4n_decrease(self.out_care.copy(), cd4_decrease_coeff.iloc[0], self.year)

    def add_new_dx(self):
        self.in_care = self.in_care.append(self.new_dx.loc[self.new_dx['h1yy'] == self.year].copy()).sort_index()

    def kill_in_care(self):
        death_prob = calculate_death_in_care_prob(self.in_care.copy(), mortality_in_care_coeff.loc[self.group_name], self.year)
        died = death_prob > np.random.rand(len(self.in_care.index))
        new_dead = self.in_care.loc[died].copy()
        new_dead['year_died'] = self.year
        self.dead_in_care = self.dead_in_care.append(new_dead.copy(), sort=False).sort_index()
        self.in_care = self.in_care.loc[~died].copy()

    def kill_out_care(self):
        death_prob = calculate_death_out_care_prob(self.out_care.copy(), mortality_out_care_coeff.loc[self.group_name], self.year)
        died = death_prob > np.random.rand(len(self.out_care.index))
        new_dead = self.out_care.loc[died].copy()
        new_dead['year_died'] = self.year
        self.dead_out_care = self.dead_out_care.append(new_dead.copy(), sort=False).sort_index()
        self.out_care = self.out_care.loc[~died].copy()
    
    def lose_to_follow_up(self):
        ltfu_prob = calculate_ltfu_prob(self.in_care.copy(), ltfu_coeff.loc[self.group_name], self.year)
        lost = ltfu_prob > np.random.rand(len(self.in_care.index))
        self.new_out_care = self.in_care.loc[lost].copy()
        self.new_out_care['sqrtcd4n_exit'] = self.new_out_care['time_varying_sqrtcd4n']
        self.new_out_care['ltfu_year'] = self.year
        self.new_out_care['n_lost'] += 1
        self.in_care = self.in_care.loc[~lost].copy()

    def reengage(self):
        prob = prob_reengage.loc[self.group_name]
        reengaged =  np.random.rand(len(self.out_care.index)) < np.full(len(self.out_care.index), prob)
        self.new_in_care = self.out_care.loc[reengaged].drop(['sqrtcd4n_exit', 'ltfu_year'], axis=1).copy()

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        self.new_in_care['init_sqrtcd4n'] = self.new_in_care['time_varying_sqrtcd4n']
        self.new_in_care['h1yy'] = self.year
        self.out_care = self.out_care.loc[~reengaged].copy()

    def append_new(self):
        self.out_care = self.out_care.append(self.new_out_care.copy()).sort_index()
        self.in_care = self.in_care.append(self.new_in_care.copy()).sort_index()

    def take_snapshot(self):
        """ Append the full population of each DataFrame with year label """
        self.in_care_snap = self.in_care_snap.append(self.in_care.assign(year=self.year, replication=self.replication))
        self.out_care_snap = self.out_care_snap.append(self.out_care.assign(year=self.year, replication=self.replication))
        self.new_in_care_snap = self.new_in_care_snap.append(self.in_care.assign(year=self.year, replication=self.replication))
        self.new_out_care_snap = self.new_out_care_snap.append(self.out_care.assign(year=self.year, replication=self.replication))

    def run_simulation(self, end):
        """ Simulate from 2009 to end """
        while(self.year <= end):
            print(self.year)

            # Everybody ages
            self.increment_age()
            
            # In care operations
            self.increase_cd4_count()                                                   # Increase cd4n in people in care
            self.add_new_dx()                                                           # Add in newly diagnosed ART initiators
            self.kill_in_care()                                                         # Kill some people in care
            self.lose_to_follow_up()                                                    # Lose some people to follow up
            
            # Out of care operations
            self.decrease_cd4_count()                                                   # Decrease cd4n in people out of care
            self.kill_out_care()                                                        # Kill some people out of care
            self.reengage()                                                             # Reengage some people out of care

            # Append changed populations to their respective DataFrames
            self.append_new()

            # Take snapshot or prepare output
            self.take_snapshot()

            # Increment year
            self.year += 1

# namedtuple for passing around output
OutputContainer = namedtuple('Output', ['final_population', 'dead_in_care', 'dead_out_care', 'in_care', 'out_care', 'new_in_care', 'new_out_care', 'new_initiators'])

###############################################################################
# Simulate Function                                                           #
###############################################################################

def simulate(replication, group_name):
    """ Run one replication of the pearl model for a given group"""

    # Create 2009 population
    population_2009 = make_pop_2009(on_art_2009.loc[group_name], mixture_2009_coeff.loc[group_name], naaccord_prop_2009.copy(), init_sqrtcd4n_coeff_2009.loc[group_name],
                                    cd4_increase_coeff.loc[group_name], group_name)

    # Simulate number of new art initiators
    art_init_sim = simulate_new_dx(new_dx.loc[group_name].copy(), new_dx_interval.loc[group_name].copy())

    # Create population of new art initiators
    new_population = make_new_population(art_init_sim.copy(), mixture_h1yy_coeff.loc[group_name].copy(), init_sqrtcd4n_coeff.loc[group_name], 
                                         cd4_increase_coeff.loc[group_name], population_2009.shape[0])

    # Initialize Pearl object
    pearl = Pearl(population_2009, new_population, group_name, replication)

    # Run simulation
    pearl.run_simulation(end=2030)

    # Prepare output
    final_population = pd.concat([pearl.in_care, pearl.out_care, pearl.dead_in_care, pearl.dead_out_care], sort=False)
    return OutputContainer(final_population, pearl.dead_in_care, pearl.dead_out_care, pearl.in_care_snap.reset_index(), 
                           pearl.out_care_snap.reset_index(), pearl.new_in_care_snap.reset_index(), pearl.new_out_care_snap.reset_index(), 
                           pearl.new_dx)

###############################################################################
# Prepare Output Function                                                     #
###############################################################################

def output_reindex(df):
    """ Helper function for reindexing output tables """
    return df.reindex( pd.MultiIndex.from_product([df.index.levels[0], np.arange(2.0, 8.0)], names=['year', 'age_cat']), fill_value=0)

#def prepare_output(final_population, dead_in_care, dead_out_care, in_care, out_care, new_in_care, new_out_care, new_initiators, group_name, replication):
def prepare_output(raw_output, group_name, replication):
    """ Take raw output and aggregate """
    
    ## Count how many times people left and tally them up
    #n_times_lost = pd.DataFrame(final_population['n_lost'].value_counts())
    #n_times_lost['pct'] = 100.0 * n_times_lost['n_lost'] / n_times_lost['n_lost'].sum()
    #n_times_lost['replication'] = replication

    ## Count of those that died out of care
    #dead_out_care_count = dead_out_care.groupby(['year_died', 'age_cat']).size().reset_index(name='n')
    #dead_out_care_count = dead_out_care_count.rename(columns={'year_died': 'year'})
    #dead_out_care_count['replication'] = replication
    #dead_out_care_count['group'] = group_name
    #
    ## Count of those that died in care
    #dead_in_care_count = dead_in_care.groupby(['year_died', 'age_cat']).size().reset_index(name='n')
    #dead_in_care_count = dead_in_care_count.rename(columns={'year_died': 'year'})
    #dead_in_care_count['replication'] = replication
    #dead_in_care_count['group'] = group_name
    #
    ## Count of those newly reengaged
    #new_in_care_count = new_in_care.groupby(['year', 'age_cat']).size().reset_index(name='n')
    #new_in_care_count['replication'] = replication
    #new_in_care_count['group'] = group_name

    ## Count of those newly lost
    #new_out_care_count = new_out_care.groupby(['year', 'age_cat']).size().reset_index(name='n')
    #new_out_care_count['replication'] = replication
    #new_out_care_count['group'] = group_name

    # Count of those in care by age_cat
    in_care_count = output_reindex(raw_output.in_care.groupby(['year', 'age_cat']).size()).reset_index(name='n')
    in_care_count['replication'] = replication
    in_care_count['group'] = group_name
    in_care_count = in_care_count.set_index(['group', 'replication', 'year', 'age_cat'])

    # Count of those out of care
    out_care_count = output_reindex(raw_output.out_care.groupby(['year', 'age_cat']).size()).reset_index(name='n')
    out_care_count['replication'] = replication
    out_care_count['group'] = group_name
    out_care_count = out_care_count.set_index(['group', 'replication', 'year', 'age_cat'])


    ## Count of new initiators
    #new_init_count = new_initiators.groupby(['h1yy']).size().reset_index(name='n')
    #new_init_count['replication'] = replication
    #new_init_count['group'] = group_name
    #
    ## Count of those in care by age
    #in_care_age = in_care.groupby(['year', 'age']).size().reset_index(name='n')
    #in_care_age['replication'] = replication
    #in_care_age['group'] = group_name

    return in_care_count

###############################################################################
# Main Function                                                               #
###############################################################################
        

def main():
    group_names = ['idu_hisp_female']
    #group_names = on_art_2009.index.values
    age_final = pd.DataFrame()
    with pd.HDFStore(out_dir + '/out.h5') as store:
        for group_name in group_names:
            for replication in range(10):
                print(replication, group_name)
                raw_output = simulate(replication, group_name)
                age_out = prepare_output(raw_output, group_name, replication)
                #age_out = prepare_output(output.final_population, output.dead_in_care, output.dead_out_care, output.in_care, 
                #               output.out_care, output.new_in_care, output.new_out_care, output.new_initiators, group_name, replication)
                age_final = age_final.append(age_out)

        store['age_final'] = age_final



main()
