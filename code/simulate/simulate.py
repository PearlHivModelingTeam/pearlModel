# Imports
import sys
import os
import numpy as np
import pandas as pd
import ray
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

def calculate_in_care_cd4n(pop, coeffs, year):
    """ Calculate time varying cd4n as a linear function of age_cat, cd4_cat, time_from_h1yy and their cross terms """
    
    # Calculate spline variables
    time_from_h1yy    =  year - pop['h1yy'].values
    time_from_h1yy_   = (np.maximum(0, time_from_h1yy - coeffs['p5' ])**2 -
                         np.maximum(0, time_from_h1yy - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    time_from_h1yy__  = (np.maximum(0, time_from_h1yy - coeffs['p35'])**2 -
                         np.maximum(0, time_from_h1yy - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    time_from_h1yy___ = (np.maximum(0, time_from_h1yy - coeffs['p65'])**2 -
                         np.maximum(0, time_from_h1yy - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])

    # Calculate time varying sqrt cd4n
    time_varying_sqrtcd4n = (coeffs['intercept_c'] +
                            (coeffs['agecat_c'] * pop['age_cat']) + 
                            (coeffs['cd4cat349_c'] * pop['cd4_cat_349']) +
                            (coeffs['cd4cat499_c'] * pop['cd4_cat_499']) +
                            (coeffs['cd4cat500_c'] * pop['cd4_cat_500']) +
                            (coeffs['time_from_h1yy_c'] * time_from_h1yy) +
                            (coeffs['_time_from_h1yy_c'] * time_from_h1yy_) +
                            (coeffs['__time_from_h1yy_c'] * time_from_h1yy__) +
                            (coeffs['___time_from_h1yy_c'] * time_from_h1yy___) +
                            (coeffs['_time_from_cd4cat349_c'] * time_from_h1yy_ * pop['cd4_cat_349']) +
                            (coeffs['_time_from_cd4cat499_c'] * time_from_h1yy_ * pop['cd4_cat_499']) +
                            (coeffs['_time_from_cd4cat500_c'] * time_from_h1yy_ * pop['cd4_cat_500']) +
                            (coeffs['__time_fro_cd4cat349_c'] * time_from_h1yy__ * pop['cd4_cat_349']) +
                            (coeffs['__time_fro_cd4cat499_c'] * time_from_h1yy__ * pop['cd4_cat_499']) +
                            (coeffs['__time_fro_cd4cat500_c'] * time_from_h1yy__ * pop['cd4_cat_500']) +
                            (coeffs['___time_fr_cd4cat349_c'] * time_from_h1yy___ * pop['cd4_cat_349']) +
                            (coeffs['___time_fr_cd4cat499_c'] * time_from_h1yy___ * pop['cd4_cat_499']) +
                            (coeffs['___time_fr_cd4cat500_c'] * time_from_h1yy___ * pop['cd4_cat_500'])).values

    return time_varying_sqrtcd4n

def calculate_out_care_cd4n(pop, coeffs, year):
    """ Calculate new time varying cd4 count for population out of care """
    time_out = year - pop['ltfu_year'].values
    diff = (coeffs['time_out_of_naaccord_c'] * time_out) + (coeffs['sqrtcd4_exit_c'] * pop['sqrtcd4n_exit'].values) +  coeffs['intercept_c'] 
    time_varying_sqrtcd4n = np.sqrt((pop['sqrtcd4n_exit'].values**2) * np.exp(diff) * 1.5)
    
    return time_varying_sqrtcd4n

def calculate_ltfu_prob(pop, coeffs, year):
    """ Calculate the probability of loss to follow up """

    # Calculate spline variables
    age    = pop['age'].values
    age_   = (np.maximum(0, age - coeffs['p5'])**2 - 
              np.maximum(0, age - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    age__  = (np.maximum(0, age - coeffs['p35'])**2 - 
              np.maximum(0, age - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    age___ = (np.maximum(0, age - coeffs['p65'])**2 - 
              np.maximum(0, age - coeffs['p95'])**2) / (coeffs['p95'] - coeffs['p5'])
    
    # Create haart_period variable
    haart_period = (pop['h1yy'].values > 2010).astype(int)
    
    # Calculate log odds
    odds = (coeffs['intercept_c'] + 
           (coeffs['age_c'] * age) + 
           (coeffs['_age_c'] * age_) + 
           (coeffs['__age_c'] * age__) + 
           (coeffs['___age_c'] * age___) + 
           (coeffs['year_c'] * year ) +
           (coeffs['sqrtcd4n_c'] * pop['init_sqrtcd4n']) +
           (coeffs['haart_period_c'] * haart_period))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob

def calculate_death_in_care_prob(pop, coeffs, year):
    """ Calculate the individual probability of dying in care """
    odds = (coeffs['intercept_est'] + 
           (coeffs['ageby10_est'] * pop['age_cat']) +
           (coeffs['sqrtcd4n_est'] * pop['init_sqrtcd4n']) +
           (coeffs['year_est'] * year) +
           (coeffs['h1yy_est'] * pop['h1yy']))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob

def calculate_death_out_care_prob(pop, coeffs, year):
    """ Calculate the individual probability of dying in care """
    odds = (coeffs['intercept_c'] + 
           (coeffs['agecat_c'] * pop['age_cat']) +
           (coeffs['tv_sqrtcd4n_c'] * pop['time_varying_sqrtcd4n']) +
           (coeffs['year_c'] * year))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob

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
    population['time_varying_sqrtcd4n'] = calculate_in_care_cd4n(population.copy(), cd4_increase_coeff, 2009)

    # Add final columns used for calculations and output
    population['n_lost'] = 0
    population['h1yy_orig'] = population['h1yy']
    population['init_sqrtcd4n_orig'] = population['init_sqrtcd4n']
    population['years_out'] = 0
    population['year_died'] = np.nan
    population['sqrtcd4n_exit'] = 0
    population['ltfu_year'] = 0

    # Set status to 1 = 'in_care'
    population['status'] = 1

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

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

    population = pd.DataFrame()
    for h1yy, coeffs in sim_coeff.groupby('h1yy'):
        grouped_pop = sim_pop(coeffs.iloc[0], art_init_sim.loc[h1yy, 'n_art_init'])
        grouped_pop['h1yy'] = h1yy
        population = pd.concat([population, grouped_pop])

    population['age'] = np.floor(population['age'])
    population['age_cat'] = np.floor(population['age'] / 10)
    population.loc[population['age_cat'] < 2, 'age_cat'] = 2
    population.loc[population['age_cat'] > 7, 'age_cat'] = 7
    
    # Add id number
    population['id'] = np.arange(pop_size_2009, (pop_size_2009 + population.index.size))

    # Pull cd4 count coefficients
    mean_intercept = init_sqrtcd4n_coeff['meanint']
    mean_slope = init_sqrtcd4n_coeff['meanslp']
    std_intercept = init_sqrtcd4n_coeff['stdint']
    std_slope = init_sqrtcd4n_coeff['stdslp']

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    population = population.set_index('h1yy')
    for h1yy, group in population.groupby(level=0):
        mu = mean_intercept + (h1yy * mean_slope)
        sigma = std_intercept + (h1yy * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.inf, mu, sigma, size)
        population.loc[h1yy, 'init_sqrtcd4n'] = sqrt_cd4n
   
    population = population.reset_index().set_index('id').sort_index()
    
    # Calculate time varying cd4 count
    population = initialize_cd4n_cat(population)
    population['time_varying_sqrtcd4n'] = population['init_sqrtcd4n']

    # Add final columns used for calculations and output
    population['n_lost'] = 0
    population['h1yy_orig'] = population['h1yy']
    population['init_sqrtcd4n_orig'] = population['init_sqrtcd4n']
    population['years_out'] = 0
    population['year_died'] = np.nan
    population['sqrtcd4n_exit'] = 0
    population['ltfu_year'] = 0

    # Set status to 0=unitiated
    population['status'] = 0

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population

###############################################################################
# Pearl Class                                                                 #
###############################################################################

class Pearl:
    def __init__(self, group_name, replication):
        self.group_name = group_name
        self.replication = replication
        self.year = 2009
        self.prob_reengage = prob_reengage.loc[self.group_name]
        
        # Simulate number of new art initiators
        art_init_sim = simulate_new_dx(new_dx.loc[group_name].copy(), new_dx_interval.loc[group_name].copy())
        
        # Create 2009 population
        self.population = make_pop_2009(on_art_2009.loc[group_name], mixture_2009_coeff.loc[group_name].copy(), naaccord_prop_2009.copy(), init_sqrtcd4n_coeff_2009.loc[group_name],
                                        cd4_increase_coeff.loc[group_name], group_name)

        # Create population of new art initiators
        self.population = self.population.append(make_new_population(art_init_sim.copy(), mixture_h1yy_coeff.loc[group_name].copy(), init_sqrtcd4n_coeff.loc[group_name], 
                                                 cd4_increase_coeff.loc[group_name], self.population.shape[0]))
        
    
        # Allow loss to follow up to occur in initial year
        self.lose_to_follow_up()
        
        # Initiate output class
        self.output_container = OutputContainer()

        #Move populations
        self.append_new()

        print(self)

        # Move to 2010
        self.year += 1
       
    def __str__(self):
        total = len(self.population.index)
        in_care = len(self.population.loc[self.population['status']==1])
        out_care = len(self.population.loc[self.population['status']==2])
        dead_in_care = len(self.population.loc[self.population['status']==5])
        dead_out_care = len(self.population.loc[self.population['status']==6])
        uninitiated = len(self.population.loc[self.population['status']==0])

        string = 'Year: ' + str(self.year) + '\n'
        string += 'Total Population Size: ' + str(total) + '\n'
        string += 'In Care Size: ' + str(in_care) + '\n'
        string += 'Out Care Size: ' + str(out_care) + '\n'
        string += 'Dead In Care Size: ' + str(dead_in_care) + '\n'
        string += 'Dead Out Care Size: ' + str(dead_out_care) + '\n'
        string += 'Uninitiated Size: ' + str(uninitiated) + '\n'

        string += 'Sizes Consistent? ' + str((total) == (in_care + out_care + dead_in_care + dead_out_care + uninitiated)) + '\n'

        return string

    def increment_age(self):
        """ Increment age for those alive in the model """
        alive_and_initiated = self.population['status'].isin([1,2])
        self.population.loc[alive_and_initiated, 'age'] += 1
        self.population['age_cat'] = np.floor(self.population['age'] / 10)
        self.population.loc[self.population['age_cat'] < 2, 'age_cat'] = 2
        self.population.loc[self.population['age_cat'] > 7, 'age_cat'] = 7
        
        # Increment number of years out
        out_care = self.population['status'] == 2
        self.population.loc[out_care, 'years_out'] += 1

    def increase_cd4_count(self):
        in_care = self.population['status'] == 1
        self.population.loc[in_care, 'time_varying_sqrtcd4n'] = calculate_in_care_cd4n(self.population.loc[in_care].copy(), 
                                                                                       cd4_increase_coeff.loc[self.group_name], self.year)

    def decrease_cd4_count(self):
        out_care = self.population['status'] == 2
        self.population.loc[out_care, 'time_varying_sqrtcd4n'] = calculate_out_care_cd4n(self.population.loc[out_care].copy(), 
                                                                                         cd4_decrease_coeff.iloc[0], self.year)

    def add_new_dx(self):
        self.population.loc[(self.population['status']==0) & (self.population['h1yy']==self.year), 'status'] = 1

    def kill_in_care(self):
        in_care = self.population['status'] == 1
        death_prob = calculate_death_in_care_prob(self.population.copy(), mortality_in_care_coeff.loc[self.group_name], self.year)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & in_care
        self.population.loc[died, 'status'] = 5
        self.population.loc[died, 'year_died'] = self.year

    def kill_out_care(self):
        out_care = self.population['status'] == 2
        death_prob = calculate_death_out_care_prob(self.population.copy(), mortality_out_care_coeff.loc[self.group_name], self.year)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & out_care
        self.population.loc[died, 'status'] = 6
        self.population.loc[died, 'year_died'] = self.year
    
    def lose_to_follow_up(self):
        in_care = self.population['status'] == 1
        ltfu_prob = calculate_ltfu_prob(self.population.copy(), ltfu_coeff.loc[self.group_name], self.year)
        lost = (ltfu_prob > np.random.rand(len(self.population.index))) & in_care
        self.population.loc[lost, 'status'] = 4
        self.population.loc[lost, 'sqrtcd4n_exit'] = self.population.loc[lost, 'time_varying_sqrtcd4n'] 
        self.population.loc[lost, 'ltfu_year'] = self.year
        self.population.loc[lost, 'n_lost'] += 1

    def reengage(self):
        out_care = self.population['status'] == 2
        reengaged = (np.random.rand(len(self.population.index)) < np.full(len(self.population.index), self.prob_reengage)) & out_care
        self.population.loc[reengaged, 'status'] = 3

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        self.population.loc[reengaged, 'init_sqrtcd4n'] = self.population.loc[reengaged, 'time_varying_sqrtcd4n']
        self.population.loc[reengaged, 'h1yy'] = self.year

    def append_new(self):
        new_in_care = self.population['status'] == 3
        new_out_care = self.population['status'] == 4

        self.population.loc[new_in_care, 'status'] = 1
        self.population.loc[new_out_care, 'status'] = 2
        
    def record_stats(self):
        pass
    
    def run_simulation(self, end):
        """ Simulate from 2009 to end """
        while(self.year <= end):
            print(self)
            
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

            # Increment year
            self.year += 1


###############################################################################
# Simulate Function                                                           #
###############################################################################

def simulate(replication, group_name):
    """ Run one replication of the pearl model for a given group"""
    # Initialize Pearl object
    pearl = Pearl(group_name, replication)

    # Run simulation
    pearl.run_simulation(end=2030)

    # Prepare output
    #final_population = pd.concat([pearl.in_care, pearl.out_care, pearl.dead_in_care, pearl.dead_out_care], sort=False)
    #return RawOutputContainer(final_population, pearl.dead_in_care, pearl.dead_out_care, pearl.in_care_snap.reset_index(), 
    #                          pearl.out_care_snap.reset_index(), pearl.new_in_care_snap.reset_index(), pearl.new_out_care_snap.reset_index(), 
    #                          pearl.new_dx)
    return True

###############################################################################
# Output Classes and Functions                                                #
###############################################################################

class OutputContainer:
    def __init__(self, n_times_lost = None, dead_in_care_count=None, dead_out_care_count=None, new_in_care_count=None, new_out_care_count = None,
                 in_care_count=None, out_care_count=None, new_init_count=None, in_care_age=None, out_care_age=None, dead_in_care_age=None,
                 dead_out_care_age=None, new_in_care_age=None, new_out_care_age=None, years_out=None, prop_ltfu=None, n_out_2010_2015=None):
        self.n_times_lost        = pd.DataFrame() if n_times_lost        is None else n_times_lost 
        self.dead_in_care_count  = pd.DataFrame() if dead_in_care_count  is None else dead_in_care_count
        self.dead_out_care_count = pd.DataFrame() if dead_out_care_count is None else dead_out_care_count 
        self.new_in_care_count   = pd.DataFrame() if new_in_care_count   is None else new_in_care_count
        self.new_out_care_count  = pd.DataFrame() if new_out_care_count  is None else new_out_care_count 
        self.in_care_count       = pd.DataFrame() if in_care_count       is None else in_care_count 
        self.out_care_count      = pd.DataFrame() if out_care_count      is None else out_care_count 
        self.new_init_count      = pd.DataFrame() if new_init_count      is None else new_init_count 
        self.in_care_age         = pd.DataFrame() if in_care_age         is None else in_care_age
        self.out_care_age        = pd.DataFrame() if out_care_age        is None else out_care_age 
        self.dead_in_care_age    = pd.DataFrame() if dead_in_care_age    is None else dead_in_care_age
        self.dead_out_care_age   = pd.DataFrame() if dead_out_care_age   is None else dead_out_care_age 
        self.new_in_care_age     = pd.DataFrame() if new_in_care_age     is None else new_in_care_age
        self.new_out_care_age    = pd.DataFrame() if new_out_care_age    is None else new_out_care_age 
        self.years_out           = pd.DataFrame() if years_out           is None else years_out 
        self.prop_ltfu           = pd.DataFrame() if prop_ltfu           is None else prop_ltfu
        self.n_out_2010_2015     = pd.DataFrame() if n_out_2010_2015     is None else n_out_2010_2015

def output_reindex(df):
    """ Helper function for reindexing output tables """
    return df.reindex( pd.MultiIndex.from_product([df.index.levels[0], np.arange(2.0, 8.0)], names=['year', 'age_cat']), fill_value=0)

def prepare_output(raw_output, group_name, replication):
    """ Take raw output and aggregate """
    
    # Count how many times people left and tally them up
    n_times_lost = pd.DataFrame(raw_output.final_population['n_lost'].value_counts()).reset_index()
    n_times_lost = n_times_lost.rename(columns={'n_lost':'n', 'index': 'n_times_lost'}) 
    n_times_lost['pct'] = 100.0 * n_times_lost['n'] / n_times_lost['n'].sum()
    n_times_lost['replication'] = replication
    n_times_lost['group'] = group_name
    
    # Count of those that died out of care by age_cat and year
    dead_in_care_count = output_reindex(raw_output.dead_in_care.groupby(['year_died', 'age_cat']).size()).reset_index(name='n')
    dead_in_care_count = dead_in_care_count.rename(columns={'year_died': 'year'})
    dead_in_care_count['replication'] = replication
    dead_in_care_count['group'] = group_name
    
    # Count of those that died out of care by age_cat and year
    dead_out_care_count = output_reindex(raw_output.dead_out_care.groupby(['year_died', 'age_cat']).size()).reset_index(name='n')
    dead_out_care_count = dead_out_care_count.rename(columns={'year_died': 'year'})
    dead_out_care_count['replication'] = replication
    dead_out_care_count['group'] = group_name
    
    # Count of those that reengaged with care by age_cat and year
    new_in_care_count = output_reindex(raw_output.new_in_care.groupby(['year', 'age_cat']).size()).reset_index(name='n')
    new_in_care_count['replication'] = replication
    new_in_care_count['group'] = group_name

    # Count of those that left care by age_cat and year
    new_out_care_count = output_reindex(raw_output.new_out_care.groupby(['year', 'age_cat']).size()).reset_index(name='n')
    new_out_care_count['replication'] = replication
    new_out_care_count['group'] = group_name
   
    # Count of those in care by age_cat and year
    in_care_count = output_reindex(raw_output.in_care.groupby(['year', 'age_cat']).size()).reset_index(name='n')
    in_care_count['replication'] = replication
    in_care_count['group'] = group_name

    # Count of those out of care by age_cat and year
    out_care_count = output_reindex(raw_output.out_care.groupby(['year', 'age_cat']).size()).reset_index(name='n')
    out_care_count['replication'] = replication
    out_care_count['group'] = group_name
    
    # Count of new initiators
    new_init_count = raw_output.new_initiators.groupby(['h1yy']).size().reset_index(name='n')
    new_init_count['replication'] = replication
    new_init_count['group'] = group_name

    # Count of those in care by age
    in_care_age = raw_output.in_care.groupby(['year', 'age']).size().reset_index(name='n')
    in_care_age['replication'] = replication
    in_care_age['group'] = group_name
    
    # Count of those out of care by age
    out_care_age = raw_output.out_care.groupby(['year', 'age']).size().reset_index(name='n')
    out_care_age['replication'] = replication
    out_care_age['group'] = group_name
    
    # Count of those dead in care by age
    dead_in_care_age = raw_output.dead_in_care.groupby(['year_died', 'age']).size().reset_index(name='n')
    dead_in_care_age = dead_in_care_age.rename(columns={'year_died': 'year'})
    dead_in_care_age['replication'] = replication
    dead_in_care_age['group'] = group_name
    
    # Count of those dead out of care by age
    dead_out_care_age = raw_output.dead_out_care.groupby(['year_died', 'age']).size().reset_index(name='n')
    dead_out_care_age = dead_out_care_age.rename(columns={'year_died': 'year'})
    dead_out_care_age['replication'] = replication
    dead_out_care_age['group'] = group_name
    
    # Count of those reengaging in care by age
    new_in_care_age = raw_output.new_in_care.groupby(['year', 'age']).size().reset_index(name='n')
    new_in_care_age['replication'] = replication
    new_in_care_age['group'] = group_name

    # Count of those newly lost to care by age
    new_out_care_age = raw_output.new_out_care.groupby(['year', 'age']).size().reset_index(name='n')
    new_out_care_age['replication'] = replication
    new_out_care_age['group'] = group_name
    
    # Count how many years spent out of care and tally
    years_out = pd.DataFrame(raw_output.final_population['years_out'].value_counts()).reset_index()    
    years_out = years_out.rename(columns={'years_out':'n', 'index':'years_out'})
    years_out['replication'] = replication
    years_out['group'] = group_name

    # Proportion of initial population and new inits 2010-2015 lost to follow up 2010-2015 TODO: 2009?
    denominator = len(raw_output.final_population.loc[raw_output.final_population['h1yy_orig'] <= 2015].index)
    numerator = len(raw_output.new_out_care.loc[(raw_output.new_out_care['year'] <= 2015) & (raw_output.new_out_care['year'] >= 2010)]['id'].unique())
    prop_ltfu = pd.DataFrame([[group_name, replication, 100.0 * numerator / denominator]], columns=['group', 'replication', 'pct'])

    # Number of unique patients lost to follow up 2010-2015
    n_out_2010_2015 = pd.DataFrame([[group_name, replication, numerator]], columns=['group', 'replication', 'n'])

    return OutputContainer(n_times_lost, dead_in_care_count, dead_out_care_count, new_in_care_count, new_out_care_count, in_care_count, 
                           out_care_count, new_init_count, in_care_age, out_care_age, dead_in_care_age, dead_out_care_age, new_in_care_age,
                           new_out_care_age, years_out, prop_ltfu, n_out_2010_2015)

def append_replications(outputs, final_output):
    for i in range(len(outputs)):
        final_output.n_times_lost        = final_output.n_times_lost.append(outputs[i].n_times_lost, ignore_index=True)
        final_output.dead_in_care_count  = final_output.dead_in_care_count.append(outputs[i].dead_in_care_count, ignore_index=True)
        final_output.dead_out_care_count = final_output.dead_out_care_count.append(outputs[i].dead_out_care_count, ignore_index=True)
        final_output.new_in_care_count   = final_output.new_in_care_count.append(outputs[i].new_in_care_count, ignore_index=True)
        final_output.new_out_care_count  = final_output.new_out_care_count.append(outputs[i].new_out_care_count, ignore_index=True)
        final_output.in_care_count       = final_output.in_care_count.append(outputs[i].in_care_count, ignore_index=True)
        final_output.out_care_count      = final_output.out_care_count.append(outputs[i].out_care_count, ignore_index=True)
        final_output.new_init_count      = final_output.new_init_count.append(outputs[i].new_init_count, ignore_index=True)
        final_output.in_care_age         = final_output.in_care_age.append(outputs[i].in_care_age, ignore_index=True)
        final_output.out_care_age        = final_output.out_care_age.append(outputs[i].out_care_age, ignore_index=True)
        final_output.dead_in_care_age    = final_output.dead_in_care_age.append(outputs[i].dead_in_care_age, ignore_index=True)
        final_output.dead_out_care_age   = final_output.dead_out_care_age.append(outputs[i].dead_out_care_age, ignore_index=True)
        final_output.new_in_care_age     = final_output.new_in_care_age.append(outputs[i].new_in_care_age, ignore_index=True)
        final_output.new_out_care_age    = final_output.new_out_care_age.append(outputs[i].new_out_care_age, ignore_index=True)
        final_output.years_out           = final_output.years_out.append(outputs[i].years_out, ignore_index=True)
        final_output.prop_ltfu           = final_output.prop_ltfu.append(outputs[i].prop_ltfu, ignore_index=True)
        final_output.n_out_2010_2015     = final_output.n_out_2010_2015.append(outputs[i].n_out_2010_2015, ignore_index=True)

    return final_output

def store_output(final_output, group_name):
    with pd.HDFStore(out_dir + '/pearl_out.h5') as store:
        store['n_times_lost']        = final_output.n_times_lost
        store['dead_in_care_count']  = final_output.dead_in_care_count
        store['dead_out_care_count'] = final_output.dead_out_care_count
        store['new_in_care_count']   = final_output.new_in_care_count
        store['new_out_care_count']  = final_output.new_out_care_count
        store['in_care_count']       = final_output.in_care_count
        store['out_care_count']      = final_output.out_care_count
        store['new_init_count']      = final_output.new_init_count
        store['in_care_age']         = final_output.in_care_age
        store['out_care_age']        = final_output.out_care_age
        store['dead_in_care_age']    = final_output.dead_in_care_age
        store['dead_out_care_age']   = final_output.dead_out_care_age
        store['new_in_care_age']     = final_output.new_in_care_age
        store['new_out_care_age']    = final_output.new_out_care_age
        store['years_out']           = final_output.years_out
        store['prop_ltfu']           = final_output.prop_ltfu
        store['n_out_2010_2015']     = final_output.n_out_2010_2015

###############################################################################
# Main Function                                                               #
###############################################################################
     

raw_output = simulate(1, 'idu_hisp_female')
