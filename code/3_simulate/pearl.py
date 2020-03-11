# Imports
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import feather


# Status Constants
UNINITIATED_USER = 0
UNINITIATED_NONUSER = 1
ART_USER = 2
ART_NONUSER = 3
REENGAGED = 4
LTFU = 5
DEAD_ART_USER = 6
DEAD_ART_NONUSER = 7


###############################################################################
# Functions                                                                   #
###############################################################################


def gather(df, key, value, cols):
    """ Implementation of gather from the tidyverse """
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt(df, id_vars, id_values, var_name, value_name)


def draw_from_trunc_norm(a, b, mu, sigma, n):
    """ Draws n values from a truncated normal with the given parameters """
    if n != 0:
        a_mod = (a - mu) / sigma
        b_mod = (b - mu) / sigma
        return stats.truncnorm.rvs(a_mod, b_mod, loc=mu, scale=sigma, size=n)
    else:
        return []


def simulate_ages(coeffs, pop_size):
    """ Draw ages from a mixed or single gaussian truncated at 18 and 85 given the coefficients and population size."""

    # Draw population size of each normal from the binomial distribution
    pop_size_1 = np.random.binomial(pop_size, coeffs.loc['lambda1', 'estimate'])
    pop_size_2 = pop_size - pop_size_1

    # Draw ages from truncated normal
    pop1 = draw_from_trunc_norm(18, 85, coeffs.loc['mu1', 'estimate'], coeffs.loc['sigma1', 'estimate'], pop_size_1)
    pop2 = draw_from_trunc_norm(18, 85, coeffs.loc['mu2', 'estimate'], coeffs.loc['sigma2', 'estimate'], pop_size_2)
    population = np.concatenate((pop1, pop2))

    # Create DataFrame
    population = pd.DataFrame(data={'age': population})

    return population


def set_cd4_cat(pop):
    """ Given initial sqrtcd4n, add columns with categories used for cd4 increase function """
    init_cd4_cat = np.select([pop['init_sqrtcd4n'].lt(np.sqrt(200.0)),
                              pop['init_sqrtcd4n'].ge(np.sqrt(200.0)) & pop['init_sqrtcd4n'].lt(np.sqrt(350.0)),
                              pop['init_sqrtcd4n'].ge(np.sqrt(350.0)) & pop['init_sqrtcd4n'].lt(np.sqrt(500.0)),
                              pop['init_sqrtcd4n'].ge(np.sqrt(500.0))],
                             [1, 2, 3, 4])
    pop['cd4_cat_349'] = (init_cd4_cat == 2).astype(int)
    pop['cd4_cat_499'] = (init_cd4_cat == 3).astype(int)
    pop['cd4_cat_500'] = (init_cd4_cat == 4).astype(int)

    return pop


def calculate_cd4_increase(pop, knots, year, coeffs, vcov, flag, rand):
    """ Calculate in care cd4 count via a linear function of time since art initiation, initial cd4 count, age
        category and cross terms"""

    # Calculate spline variables
    pop['time_from_h1yy'] = year - pop['h1yy']
    pop['time_from_h1yy_'] = (np.maximum(0, pop['time_from_h1yy'] - knots['p5']) ** 2 -
                              np.maximum(0, pop['time_from_h1yy'] - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    pop['time_from_h1yy__'] = (np.maximum(0, pop['time_from_h1yy'] - knots['p35']) ** 2 -
                               np.maximum(0, pop['time_from_h1yy'] - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    pop['time_from_h1yy___'] = (np.maximum(0, pop['time_from_h1yy'] - knots['p65']) ** 2 -
                                np.maximum(0, pop['time_from_h1yy'] - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])

    # Create cross term variables
    pop['timecd4cat349_'] = pop['time_from_h1yy_'] * pop['cd4_cat_349']
    pop['timecd4cat499_'] = pop['time_from_h1yy_'] * pop['cd4_cat_499']
    pop['timecd4cat500_'] = pop['time_from_h1yy_'] * pop['cd4_cat_500']

    pop['timecd4cat349__'] = pop['time_from_h1yy__'] * pop['cd4_cat_349']
    pop['timecd4cat499__'] = pop['time_from_h1yy__'] * pop['cd4_cat_499']
    pop['timecd4cat500__'] = pop['time_from_h1yy__'] * pop['cd4_cat_500']

    pop['timecd4cat349___'] = pop['time_from_h1yy___'] * pop['cd4_cat_349']
    pop['timecd4cat499___'] = pop['time_from_h1yy___'] * pop['cd4_cat_499']
    pop['timecd4cat500___'] = pop['time_from_h1yy___'] * pop['cd4_cat_500']

    # Create numpy matrix
    pop_matrix = pop[['intercept', 'time_from_h1yy', 'time_from_h1yy_', 'time_from_h1yy__', 'time_from_h1yy___',
                      'cd4_cat_349', 'cd4_cat_499', 'cd4_cat_500', 'age_cat', 'timecd4cat349_', 'timecd4cat499_',
                      'timecd4cat500_', 'timecd4cat349__', 'timecd4cat499__', 'timecd4cat500__',
                      'timecd4cat349___', 'timecd4cat499___', 'timecd4cat500___']].to_numpy()

    # Perform matrix multiplication
    new_cd4 = np.matmul(pop_matrix, coeffs)

    if flag:
        # Calculate variance of prediction using matrix multiplication
        se = np.sqrt(np.sum(np.matmul(pop_matrix, vcov) * pop_matrix, axis=1))
        low = new_cd4 - 1.96 * se
        high = new_cd4 + 1.96 * se
        new_cd4 = (rand * (high - low)) + low

    return new_cd4

def calculate_cd4_decrease(pop, coeffs, flag, vcov, rand):
    pop['time_out'] = pop['year'] - pop['ltfu_year']
    pop_matrix = pop[['intercept', 'time_out', 'sqrtcd4n_exit']].to_numpy()
    diff = np.matmul(pop_matrix, coeffs)

    if flag:
        se = np.sqrt(np.sum(np.matmul(pop_matrix, vcov) * pop_matrix, axis=1))
        low = diff - 1.96 * se
        high = diff + 1.96 * se
        diff = (rand * (high - low)) + low

    new_cd4 = np.sqrt((pop['sqrtcd4n_exit'].to_numpy() ** 2)*np.exp(diff) * 1.5 )
    return new_cd4

def create_comorbidity_pop_matrix(pop, condition):
    pop['time_since_art'] = pop['year'] - pop['h1yy_orig']
    pop['out_care'] = (pop['status'] == ART_NONUSER).astype(int)

    if condition=='anxiety':
        return pop[['age', 'init_sqrtcd4n_orig', 'depression', 'time_since_art', 'hcv', 'intercept', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='depression':
        return pop[['age', 'anxiety', 'init_sqrtcd4n_orig', 'time_since_art', 'hcv', 'intercept', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='ckd':
        return pop[['age', 'anxiety', 'init_sqrtcd4n_orig', 'diabetes', 'depression', 'time_since_art', 'hcv', 'hypertension', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='lipid':
        return pop[['age', 'anxiety', 'init_sqrtcd4n_orig', 'ckd', 'diabetes', 'depression', 'time_since_art', 'hcv', 'hypertension', 'intercept', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='diabetes':
        return pop[['age', 'anxiety', 'init_sqrtcd4n_orig', 'ckd', 'depression', 'time_since_art', 'hcv', 'hypertension', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='hypertension':
        return pop[['age', 'anxiety', 'init_sqrtcd4n_orig', 'ckd', 'diabetes', 'depression', 'time_since_art', 'hcv', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy()

def create_ltfu_pop_matrix(pop, knots):
    """ Create the population matrix for use in calculating probability of loss to follow up"""

    age = pop['age'].values
    pop['age_']   = (np.maximum(0, age - knots['p5']) ** 2 -
                     np.maximum(0, age - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    pop['age__']  = (np.maximum(0, age - knots['p35']) ** 2 -
                     np.maximum(0, age - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    pop['age___'] = (np.maximum(0, age - knots['p65']) ** 2 -
                     np.maximum(0, age - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])

    pop['haart_period'] = (pop['h1yy'].values > 2010).astype(int)
    return pop[['intercept', 'age', 'age_', 'age__', 'age___', 'year', 'init_sqrtcd4n', 'haart_period']].to_numpy()

def calculate_prob(pop, coeffs, flag, vcov, rand):
    """ Calculate the individual probability from logistic regression """

    log_odds = np.matmul(pop, coeffs)
    if flag:
        # Calculate variance of prediction using matrix multiplication
        se = np.sqrt(np.sum(np.matmul(pop, vcov) * pop, axis=1))
        low = log_odds - 1.96 * se
        high = log_odds + 1.96 * se
        log_odds = (rand * (high - low)) + low

    # Convert to probability
    prob = np.exp(log_odds) / (1.0 + np.exp(log_odds))
    return prob

def simulate_new_dx(new_dx_param, linkage_to_care):
    """ Draw number of new diagnoses from a uniform distribution between upper and lower bounds. Calculate number of
    new art initiators by assuming 75% link in the first year, then another 10% over the next three years. Assume 75%
    of these initiate art """

    new_dx = new_dx_param.copy()

    # Draw new dx from a uniform distribution between upper and lower for 2016-2030
    new_dx['n_dx'] = new_dx['lower'] + (new_dx['upper'] - new_dx['lower']) * np.random.uniform()

    # Only a proportion of new diagnoses link to care and 40% of the remaining link in the next 3 years
    new_dx['unlinked'] = new_dx['n_dx'] * (1 - linkage_to_care['link_prob'])
    new_dx['gardner_per_year'] = new_dx['unlinked'] * 0.4 / 3.0
    new_dx['year0'] = new_dx['n_dx'] * linkage_to_care['link_prob']
    new_dx['year1'] = new_dx['gardner_per_year'].shift(1, fill_value=0)
    new_dx['year2'] = new_dx['gardner_per_year'].shift(2, fill_value=0)
    new_dx['year3'] = new_dx['gardner_per_year'].shift(3, fill_value=0)
    new_dx['total_linked'] = new_dx['year0'] + new_dx['year1'] + new_dx['year2'] + new_dx['year3']

    # Proportion of those linked to care start ART
    new_dx['art_users'] = (new_dx['total_linked'] * linkage_to_care['art_prob']).astype(int)
    new_dx['art_nonusers'] = (new_dx['total_linked'] * (1 - linkage_to_care['art_prob'])).astype(int)

    # Count those not starting art 2006 - 2009 as initial ART nonusers
    n_initial_nonusers = new_dx.loc[np.arange(2006, 2010), 'art_nonusers'].sum()
    n_initial_nonusers = 0

    # Compile list of number of new agents to be introduced in the model
    new_agents = new_dx.loc[np.arange(2010, 2031), ['art_users', 'art_nonusers']]
    new_agents['total'] = new_agents['art_users'] + new_agents['art_nonusers']

    return n_initial_nonusers, new_agents

def make_pop_2009(parameters, n_initial_nonusers, out_dir, group_name, replication):
    """ Create initial 2009 population. Draw ages from a mixed normal distribution truncated at 18 and 85. h1yy is
    assigned using proportions from NA-ACCORD data. Finally, sqrt cd4n is drawn from a 0-truncated normal for each
    h1yy """

    # Draw ages from the truncated mixed gaussian
    pop_size = parameters.on_art_2009[0].astype('int') + n_initial_nonusers
    population = simulate_ages(parameters.age_in_2009, pop_size)

    #print(parameters.on_art_2009[0].astype('int'))
    #print(n_initial_nonusers)

    # Create age categories
    population['age'] = np.floor(population['age'])
    population['age_cat'] = np.floor(population['age'] / 10)
    population.loc[population['age_cat'] > 7, 'age_cat'] = 7
    population['id'] = range(population.index.size)
    population = population.sort_values('age')
    population = population.set_index(['age_cat', 'id'])

    # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
    for age_cat, grouped in population.groupby('age_cat'):
        if parameters.h1yy_by_age_2009.index.isin([(group_name, age_cat, 2000.0)]).any():
            h1yy_data = parameters.h1yy_by_age_2009.loc[(group_name, age_cat)]
        else:  # replace missing data
            h1yy_data = parameters.h1yy_by_age_2009.loc[('msm_white_male', age_cat)]
        population.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data.index.values, size=grouped.shape[0],
                                                           p=h1yy_data.pct.values)

    # Pull cd4 count coefficients
    mean_intercept = parameters.cd4n_by_h1yy_2009['meanint']
    mean_slope = parameters.cd4n_by_h1yy_2009['meanslp']
    std_intercept = parameters.cd4n_by_h1yy_2009['stdint']
    std_slope = parameters.cd4n_by_h1yy_2009['stdslp']

    # Reindex for group operation
    population['h1yy'] = population['h1yy'].astype(int)
    population = population.reset_index().set_index(['h1yy', 'id']).sort_index()

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    for h1yy, group in population.groupby(level=0):
        mu = mean_intercept + (h1yy * mean_slope)
        sigma = std_intercept + (h1yy * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size)
        population.loc[(h1yy,), 'init_sqrtcd4n'] = sqrt_cd4n

    population = population.reset_index().set_index('id').sort_index()

    # Toss out age_cat < 2
    population.loc[population['age_cat'] < 2, 'age_cat'] = 2

    # Calculate time varying cd4 count
    population = set_cd4_cat(population)
    population['h1yy_orig'] = population['h1yy']
    population['init_sqrtcd4n_orig'] = population['init_sqrtcd4n']

    #cd4n_out = population[['init_sqrtcd4n', 'h1yy']].assign(group=group_name, replication=replication).copy().reset_index()
    #print(cd4n_out)
    #cd4n_out.to_feather(f'{out_dir}/{group_name}_cd4n_out_{replication}.feather')

    # Add final columns used for calculations and output
    population['n_lost'] = 0
    population['years_out'] = 0
    population['year_died'] = np.nan
    population['sqrtcd4n_exit'] = 0
    population['ltfu_year'] = 0
    population['intercept'] = 1.0
    population['year'] = 2009

    population['time_varying_sqrtcd4n'] = calculate_cd4_increase(population.copy(), parameters.cd4_increase_knots, 2009,
                                                                 parameters.cd4_increase.to_numpy(),
                                                                 parameters.cd4_increase_vcov.to_numpy(),
                                                                 parameters.cd4_increase_flag,
                                                                 parameters.cd4_increase_rand)
    # Set status
    population['status'] = -1
    population.loc[:n_initial_nonusers, 'status'] = ART_NONUSER
    population.loc[n_initial_nonusers:, 'status'] = ART_USER
    population.loc[:n_initial_nonusers, 'sqrtcd4n_exit'] = population.loc[n_initial_nonusers, 'time_varying_sqrtcd4n']
    population.loc[:n_initial_nonusers, 'ltfu_year'] = 2009
    population.loc[:n_initial_nonusers, 'n_lost'] += 1

    if parameters.comorbidity_flag:
        # Stage 0 comorbidities
        population['smoking'] = (np.random.rand(len(population.index)) < parameters.smoking_prev_users.values).astype(int)
        population['hcv'] = (np.random.rand(len(population.index)) < parameters.hcv_prev_users.values).astype(int)

        # Stage 1 comorbidities
        population['anxiety'] = (np.random.rand(len(population.index)) < parameters.anxiety_prev_users.values).astype(int)
        population['depression'] = (np.random.rand(len(population.index)) < parameters.depression_prev_users.values).astype(int)

        # Stage 2 comorbidities
        population['ckd'] = (np.random.rand(len(population.index)) < parameters.ckd_prev_users.values).astype(int)
        population['lipid'] = (np.random.rand(len(population.index)) < parameters.lipid_prev_users.values).astype(int)
        population['diabetes'] = (np.random.rand(len(population.index)) < parameters.diabetes_prev_users.values).astype(int)
        population['hypertension'] = (np.random.rand(len(population.index)) < parameters.hypertension_prev_users.values).astype(int)

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population

def make_new_population(parameters, n_new_agents, pop_size_2009, out_dir, group_name, replication):
    """ Draw ages for new art initiators """

    # Draw a random value between predicted and 2018 predicted value for years greater than 2018
    rand = np.random.rand(len(parameters.age_by_h1yy.index))
    parameters.age_by_h1yy['estimate'] = rand * (parameters.age_by_h1yy['high_value'] - parameters.age_by_h1yy['low_value']) + parameters.age_by_h1yy['low_value']

    #out = parameters.age_by_h1yy[['estimate']].assign(group=group_name, replication=replication).reset_index()

    # Create population
    population = pd.DataFrame()
    for h1yy in parameters.age_by_h1yy.index.levels[0]:
        n_users = n_new_agents.loc[h1yy, 'art_users']
        n_nonusers = n_new_agents.loc[h1yy, 'art_nonusers']
        grouped_pop = simulate_ages(parameters.age_by_h1yy.loc[h1yy], n_users + n_nonusers)
        grouped_pop['h1yy'] = h1yy
        grouped_pop['status'] = -1
        grouped_pop.loc[:n_nonusers, 'status'] = UNINITIATED_NONUSER
        grouped_pop.loc[n_nonusers:, 'status'] = UNINITIATED_USER
        population = pd.concat([population, grouped_pop])

    population['age'] = np.floor(population['age'])
    population['age_cat'] = np.floor(population['age'] / 10)
    population.loc[population['age_cat'] < 2, 'age_cat'] = 2
    population.loc[population['age_cat'] > 7, 'age_cat'] = 7

    # Add id number
    population['id'] = np.arange(pop_size_2009, (pop_size_2009 + population.index.size))

    # Pull cd4 count coefficients
    mean_intercept = parameters.cd4n_by_h1yy['meanint']
    mean_slope = parameters.cd4n_by_h1yy['meanslp']
    std_intercept = parameters.cd4n_by_h1yy['stdint']
    std_slope = parameters.cd4n_by_h1yy['stdslp']

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    population = population.set_index('h1yy')
    for h1yy, group in population.groupby(level=0):
        h1yy_mod = np.where(h1yy >= 2020, 2020, h1yy)  # Pin coefficients at 2020
        mu = mean_intercept + (h1yy_mod * mean_slope)
        sigma = std_intercept + (h1yy_mod * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size)
        population.loc[h1yy, 'init_sqrtcd4n'] = sqrt_cd4n

    population = population.reset_index().set_index('id').sort_index()

    # Calculate time varying cd4 count
    population = set_cd4_cat(population)
    population['h1yy_orig'] = population['h1yy']
    population['init_sqrtcd4n_orig'] = population['init_sqrtcd4n']
    population['time_varying_sqrtcd4n'] = population['init_sqrtcd4n']

    # Add final columns used for calculations and output
    population['n_lost'] = 0
    population['years_out'] = 0
    population['year_died'] = np.nan
    population['sqrtcd4n_exit'] = 0
    population['ltfu_year'] = 0
    population['intercept'] = 1.0
    population['year'] = population['h1yy_orig']

    # Set initial data for nonusers
    uninitiated_nonusers = population['status'] == UNINITIATED_NONUSER
    population.loc[uninitiated_nonusers, 'sqrtcd4n_exit'] = population.loc[uninitiated_nonusers, 'time_varying_sqrtcd4n']
    population.loc[uninitiated_nonusers, 'ltfu_year'] = population.loc[uninitiated_nonusers, 'h1yy_orig']
    population.loc[uninitiated_nonusers, 'n_lost'] += 1

    if parameters.comorbidity_flag:
        # Stage 0 comorbidities
        population['smoking'] = (np.random.rand(len(population.index)) < parameters.smoking_prev_inits.values).astype(int)
        population['hcv'] = (np.random.rand(len(population.index)) < parameters.hcv_prev_inits.values).astype(int)

        # Stage 1 comorbidities
        population['anxiety'] = (np.random.rand(len(population.index)) < parameters.anxiety_prev_inits.values).astype(int)
        population['depression'] = (np.random.rand(len(population.index)) < parameters.depression_prev_inits.values).astype(int)

        # Stage 2 comorbidities
        population['ckd'] = (np.random.rand(len(population.index)) < parameters.ckd_prev_inits.values).astype(int)
        population['lipid'] = (np.random.rand(len(population.index)) < parameters.lipid_prev_inits.values).astype(int)
        population['diabetes'] = (np.random.rand(len(population.index)) < parameters.diabetes_prev_inits.values).astype(int)
        population['hypertension'] = (np.random.rand(len(population.index)) < parameters.hypertension_prev_inits.values).astype(int)

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population

def create_multimorbidity_stats(pop):
    # Encode multimorbidity as 8 bit integer
    df = pop[['age_cat', 'smoking', 'hcv', 'anxiety', 'depression', 'ckd', 'lipid', 'diabetes', 'hypertension']].copy()
    df['multimorbidity'] = (
            df['smoking'].map(str) + df['hcv'].map(str) + df['anxiety'].map(str) + df['depression'].map(str)
            + df['ckd'].map(str) + df['lipid'].map(str) + df['diabetes'].map(str) + df['hypertension'].map(
        str)).apply(int, base=2)

    # Count how many people have each unique set of comorbidities
    df = df.groupby(['age_cat', 'multimorbidity']).size()
    index = pd.MultiIndex.from_product([df.index.levels[0], range(256)], names=['age_cat', 'multimorbidity'])
    df = df.reindex(index=index, fill_value=0).reset_index(name='n')

    return (df)


###############################################################################
# Parameter and Statistics Classes                                            #
###############################################################################

class Parameters:
    def __init__(self, path, group_name, comorbidity_flag, dx_reduce_flag, sensitivity_analysis_list):
        # Unpack Sensitivity Analysis List
        age_in_2009_flag=sensitivity_analysis_list[0]
        mortality_in_care_flag=sensitivity_analysis_list[1]
        mortality_out_care_flag=sensitivity_analysis_list[2]
        loss_to_follow_up_flag=sensitivity_analysis_list[3]
        cd4_increase_flag=sensitivity_analysis_list[4]
        cd4_decrease_flag=sensitivity_analysis_list[5]

        with pd.HDFStore(path) as store:
            # 2009 population
            self.on_art_2009 = store['on_art_2009'].loc[group_name]
            self.age_in_2009 = store['age_in_2009'].loc[group_name]
            self.age_in_2009_rand = np.random.rand(5)
            self.h1yy_by_age_2009 = store['h1yy_by_age_2009']
            self.cd4n_by_h1yy_2009 = store['cd4n_by_h1yy_2009'].loc[group_name]
            if age_in_2009_flag:
                self.age_in_2009['estimate'] = (self.age_in_2009_rand * (self.age_in_2009['conf_high'] - self.age_in_2009['conf_low'])) + self.age_in_2009['conf_low']

            # New ART initiators
            self.new_dx = store['new_dx'].loc[group_name]
            if dx_reduce_flag:
                self.new_dx = store['new_dx_ehe'].loc[group_name]
            else:
                self.new_dx = store['new_dx'].loc[group_name]
            self.linkage_to_care = store['linkage_to_care'].loc[group_name]
            self.age_by_h1yy = store['age_by_h1yy'].loc[group_name]
            self.cd4n_by_h1yy = store['cd4n_by_h1yy'].loc[group_name]

            # Mortality In Care
            self.mortality_in_care = store['mortality_in_care'].loc[group_name]
            self.mortality_in_care_vcov = store['mortality_in_care_vcov'].loc[group_name]
            self.mortality_in_care_flag = mortality_in_care_flag
            self.mortality_in_care_rand = np.random.rand()

            # Mortality Out Of Care
            self.mortality_out_care = store['mortality_out_care'].loc[group_name]
            self.mortality_out_care_vcov = store['mortality_out_care_vcov'].loc[group_name]
            self.mortality_out_care_flag = mortality_out_care_flag
            self.mortality_out_care_rand = np.random.rand()

            # Loss To Follow Up
            self.loss_to_follow_up = store['loss_to_follow_up'].loc[group_name]
            self.loss_to_follow_up_vcov = store['loss_to_follow_up_vcov'].loc[group_name]
            self.loss_to_follow_up_flag = loss_to_follow_up_flag
            self.loss_to_follow_up_rand = np.random.rand()
            self.ltfu_knots = store['ltfu_knots'].loc[group_name]

            # Cd4 Increase
            self.cd4_increase = store['cd4_increase'].loc[group_name]
            self.cd4_increase_vcov = store['cd4_increase_vcov'].loc[group_name]
            self.cd4_increase_flag = cd4_increase_flag
            self.cd4_increase_rand = np.random.rand()
            self.cd4_increase_knots = store['cd4_increase_knots'].loc[group_name]

            # Cd4 Decrease
            self.cd4_decrease = store['cd4_decrease'].loc['all']
            self.cd4_decrease_vcov = store['cd4_decrease_vcov']
            self.cd4_decrease_flag = cd4_decrease_flag
            self.cd4_decrease_rand = np.random.rand()

            # Reengagement probability
            self.prob_reengage = store['prob_reengage'].loc[group_name]

            # Stage 0 Comorbidities
            self.hcv_prev_users = store['hcv_prev_users'].loc[group_name]
            self.smoking_prev_users = store['smoking_prev_users'].loc[group_name]

            self.hcv_prev_inits = store['hcv_prev_inits'].loc[group_name]
            self.smoking_prev_inits = store['smoking_prev_inits'].loc[group_name]

            # Stage 1 Comorbidities
            self.anxiety_prev_users = store['anxiety_prev_users'].loc[group_name]
            self.depression_prev_users = store['depression_prev_users'].loc[group_name]

            self.anxiety_prev_inits = store['anxiety_prev_inits'].loc[group_name]
            self.depression_prev_inits = store['depression_prev_inits'].loc[group_name]

            self.anxiety_coeff = store['anxiety_coeff'].loc[group_name]
            self.depression_coeff = store['depression_coeff'].loc[group_name]

            # Stage 2 Comorbidities
            self.ckd_prev_users = store['ckd_prev_users'].loc[group_name]
            self.lipid_prev_users = store['lipid_prev_users'].loc[group_name]
            self.diabetes_prev_users = store['diabetes_prev_users'].loc[group_name]
            self.hypertension_prev_users = store['hypertension_prev_users'].loc[group_name]

            self.ckd_prev_inits = store['ckd_prev_inits'].loc[group_name]
            self.lipid_prev_inits = store['lipid_prev_inits'].loc[group_name]
            self.diabetes_prev_inits = store['diabetes_prev_inits'].loc[group_name]
            self.hypertension_prev_inits = store['hypertension_prev_inits'].loc[group_name]

            self.ckd_coeff = store['ckd_coeff'].loc[group_name]
            self.lipid_coeff = store['lipid_coeff'].loc[group_name]
            self.diabetes_coeff = store['diabetes_coeff'].loc[group_name]
            self.hypertension_coeff = store['hypertension_coeff'].loc[group_name]

            # Comortality
            self.comorbidity_flag = comorbidity_flag
            if self.comorbidity_flag:
                self.mortality_in_care = store['mortality_in_care_co'].loc[group_name]
                self.mortality_out_care = store['mortality_out_care_co'].loc[group_name]


class Statistics:
    def __init__(self):
        self.in_care_count = pd.DataFrame()
        self.in_care_age = pd.DataFrame()
        self.out_care_count = pd.DataFrame()
        self.out_care_age = pd.DataFrame()
        self.reengaged_count = pd.DataFrame()
        self.reengaged_age = pd.DataFrame()
        self.ltfu_count = pd.DataFrame()
        self.ltfu_age = pd.DataFrame()
        self.died_in_care_count = pd.DataFrame()
        self.died_in_care_age = pd.DataFrame()
        self.died_out_care_count = pd.DataFrame()
        self.died_out_care_age = pd.DataFrame()
        self.new_init_count = pd.DataFrame()
        self.new_init_age = pd.DataFrame()
        self.years_out = pd.DataFrame()
        self.n_times_lost = pd.DataFrame()
        self.unique_out_care_ids = set()

        # Multimorbidity
        self.multimorbidity_in_care = pd.DataFrame()
        self.multimorbidity_dead = pd.DataFrame()

        # Incidence
        self.anxiety_incidence = pd.DataFrame()
        self.depression_incidence = pd.DataFrame()

        self.ckd_incidence = pd.DataFrame()
        self.lipid_incidence = pd.DataFrame()
        self.diabetes_incidence = pd.DataFrame()
        self.hypertension_incidence = pd.DataFrame()


def output_reindex(df):
    """ Helper function for reindexing output tables """
    return df.reindex(pd.MultiIndex.from_product([df.index.levels[0], np.arange(2.0, 8.0)], names=['year', 'age_cat']),
                      fill_value=0)


###############################################################################
# Pearl Class                                                                 #
###############################################################################

class Pearl:
    def __init__(self, parameters, group_name, replication, verbose=False):
        self.out_dir = os.path.realpath(f'{os.getcwd()}/../../out/raw')
        self.group_name = group_name
        self.replication = replication
        self.verbose = verbose
        self.year = 2009
        self.parameters = parameters

        # Simulate number of new art initiators
        n_initial_nonusers, n_new_agents = simulate_new_dx(parameters.new_dx, parameters.linkage_to_care)

        # Create 2009 population
        self.population = make_pop_2009(parameters, n_initial_nonusers, self.out_dir, self.group_name, self.replication)

        # Create population of new art initiators
        self.population = self.population.append(
            make_new_population(parameters, n_new_agents, len(self.population.index), self.out_dir, self.group_name, self.replication))

        # Initiate output class
        self.stats = Statistics()

        # First recording of stats
        self.record_stats()

        # Print populations
        if self.verbose:
            print(self)

        # Move to 2010
        self.year += 1

        # Run
        self.run()

    def __str__(self):
        total = len(self.population.index)
        in_care = len(self.population.loc[self.population['status'] == ART_USER])
        out_care = len(self.population.loc[self.population['status'] == ART_NONUSER])
        dead_in_care = len(self.population.loc[self.population['status'] == DEAD_ART_USER])
        dead_out_care = len(self.population.loc[self.population['status'] == DEAD_ART_NONUSER])
        uninitiated = len(self.population.loc[self.population['status'].isin([UNINITIATED_USER, UNINITIATED_NONUSER])])

        string = 'Year End: ' + str(self.year) + '\n'
        string += 'Total Population Size: ' + str(total) + '\n'
        string += 'In Care Size: ' + str(in_care) + '\n'
        string += 'Out Care Size: ' + str(out_care) + '\n'
        string += 'Dead In Care Size: ' + str(dead_in_care) + '\n'
        string += 'Dead Out Care Size: ' + str(dead_out_care) + '\n'
        string += 'Uninitiated Size: ' + str(uninitiated) + '\n'
        return string

    def increment_age(self):
        """ Increment age for those alive in the model """
        self.population['year'] = self.year
        alive_and_initiated = self.population['status'].isin([ART_USER, ART_NONUSER])
        self.population.loc[alive_and_initiated, 'age'] += 1
        self.population['age_cat'] = np.floor(self.population['age'] / 10)
        self.population.loc[self.population['age_cat'] < 2, 'age_cat'] = 2
        self.population.loc[self.population['age_cat'] > 7, 'age_cat'] = 7

        # Increment number of years out
        out_care = self.population['status'] == ART_NONUSER
        self.population.loc[out_care, 'years_out'] += 1

    def increase_cd4_count(self):
        in_care = self.population['status'] == ART_USER
        self.population.loc[in_care, 'time_varying_sqrtcd4n'] = calculate_cd4_increase(
            self.population.loc[in_care].copy(),
            self.parameters.cd4_increase_knots,
            self.year,
            self.parameters.cd4_increase.to_numpy(),
            self.parameters.cd4_increase_vcov.to_numpy(),
            self.parameters.cd4_increase_flag,
            self.parameters.cd4_increase_rand)


    def decrease_cd4_count(self):
        out_care = self.population['status'] == ART_NONUSER
        self.population.loc[out_care, 'time_varying_sqrtcd4n'] = calculate_cd4_decrease(
            self.population.loc[out_care].copy(),
            self.parameters.cd4_decrease.to_numpy(),
            self.parameters.cd4_decrease_flag,
            self.parameters.cd4_decrease_vcov.to_numpy(),
            self.parameters.cd4_decrease_rand)


    def add_new_dx(self):
        new_user = (self.population['status'] == UNINITIATED_USER) & (self.population['h1yy_orig'] == self.year)
        self.population.loc[new_user, 'status'] = ART_USER

        new_nonuser = (self.population['status'] == UNINITIATED_NONUSER) & (self.population['h1yy_orig'] == self.year)
        self.population.loc[new_nonuser, 'status'] = ART_NONUSER
        self.population.loc[new_nonuser, 'sqrtcd4n_exit'] = self.population.loc[new_nonuser, 'time_varying_sqrtcd4n']
        self.population.loc[new_nonuser, 'ltfu_year'] = self.year
        self.population.loc[new_nonuser, 'n_lost'] += 1

    def kill_in_care(self):
        in_care = self.population['status'] == ART_USER
        coeff_matrix = self.parameters.mortality_in_care.to_numpy()
        if self.parameters.comorbidity_flag:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'init_sqrtcd4n', 'h1yy', 'smoking', 'hcv', 'anxiety', 'depression', 'hypertension', 'diabetes', 'ckd', 'lipid']].to_numpy()
        else:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'init_sqrtcd4n', 'h1yy']].to_numpy()
        vcov_matrix = self.parameters.mortality_in_care_vcov.to_numpy()
        death_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.mortality_in_care_flag,
                                    vcov_matrix, self.parameters.mortality_in_care_rand)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & in_care
        self.population.loc[died, 'status'] = DEAD_ART_USER
        self.population.loc[died, 'year_died'] = self.year

    def kill_out_care(self):
        out_care = self.population['status'] == ART_NONUSER
        coeff_matrix = self.parameters.mortality_out_care.to_numpy()
        if self.parameters.comorbidity_flag:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'time_varying_sqrtcd4n', 'smoking', 'hcv', 'anxiety', 'depression', 'hypertension', 'diabetes', 'ckd', 'lipid']].to_numpy()
        else:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'time_varying_sqrtcd4n']].to_numpy()
        vcov_matrix = self.parameters.mortality_out_care_vcov.to_numpy()
        death_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.mortality_out_care_flag,
                                    vcov_matrix, self.parameters.mortality_out_care_rand)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & out_care
        self.population.loc[died, 'status'] = DEAD_ART_NONUSER
        self.population.loc[died, 'year_died'] = self.year

    def lose_to_follow_up(self):
        in_care = self.population['status'] == ART_USER
        coeff_matrix = self.parameters.loss_to_follow_up.to_numpy()
        vcov_matrix = self.parameters.loss_to_follow_up_vcov.to_numpy()
        pop_matrix = create_ltfu_pop_matrix(self.population.copy(), self.parameters.ltfu_knots)
        ltfu_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.loss_to_follow_up_flag,
                                   vcov_matrix, self.parameters.loss_to_follow_up_rand)
        lost = (ltfu_prob > np.random.rand(len(self.population.index))) & in_care
        self.population.loc[lost, 'status'] = LTFU
        self.population.loc[lost, 'sqrtcd4n_exit'] = self.population.loc[lost, 'time_varying_sqrtcd4n']
        self.population.loc[lost, 'ltfu_year'] = self.year
        self.population.loc[lost, 'n_lost'] += 1

    def reengage(self):
        out_care = self.population['status'] == ART_NONUSER
        reengaged = (np.random.rand(len(self.population.index)) < np.full(len(self.population.index),
                                                                          self.parameters.prob_reengage)) & out_care
        self.population.loc[reengaged, 'status'] = REENGAGED

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        self.population = set_cd4_cat(self.population)
        self.population.loc[reengaged, 'init_sqrtcd4n'] = self.population.loc[reengaged, 'time_varying_sqrtcd4n']
        self.population.loc[reengaged, 'h1yy'] = self.year

    def append_new(self):
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU

        self.population.loc[reengaged, 'status'] = ART_USER
        self.population.loc[ltfu, 'status'] = ART_NONUSER

    def apply_stage_1(self):
        in_care = self.population['status'] == ART_USER
        out_care = self.population['status'] == ART_NONUSER

        # Use matrix multiplication to calculate probability of anxiety incidence
        anxiety_coeff_matrix = self.parameters.anxiety_coeff.to_numpy()
        anxiety_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition ='anxiety')
        anxiety_prob = calculate_prob(anxiety_pop_matrix, anxiety_coeff_matrix, False, 0, 0)
        anxiety_rand = anxiety_prob > np.random.rand(len(self.population.index))
        old_anxiety = self.population['anxiety']
        new_anxiety = anxiety_rand & (in_care | out_care) & ~old_anxiety

        # Save incidence
        denominator = (self.population.loc[in_care | out_care].groupby(['age_cat']).size()
                       .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='N'))['N']
        anxiety_incidence = (self.population.loc[new_anxiety].groupby(['age_cat']).size()
                         .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        anxiety_incidence['N'] = denominator

        self.stats.anxiety_incidence = self.stats.anxiety_incidence.append(anxiety_incidence)

        # Set variables
        self.population['anxiety'] = (old_anxiety | new_anxiety).astype(int)

        # Use matrix multiplication to calculate probability of depression incidence
        depression_coeff_matrix = self.parameters.depression_coeff.to_numpy()
        depression_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='depression')
        depression_prob = calculate_prob(depression_pop_matrix, depression_coeff_matrix, False, 0, 0)
        depression_rand = depression_prob > np.random.rand(len(self.population.index))
        old_depression = self.population['depression']
        new_depression = depression_rand & (in_care | out_care) & ~old_depression

        # Save incidence
        depression_incidence = (self.population.loc[new_depression].groupby(['age_cat']).size()
                         .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        depression_incidence['N'] = denominator


        self.stats.depression_incidence = self.stats.depression_incidence.append(depression_incidence)

        # Set variables
        self.population['depression'] = (old_depression | new_depression).astype(int)

    def apply_stage_2(self):
        in_care = self.population['status'] == ART_USER
        out_care = self.population['status'] == ART_NONUSER

        # ckd
        ckd_coeff_matrix = self.parameters.ckd_coeff.to_numpy()
        ckd_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='ckd')
        ckd_prob = calculate_prob(ckd_pop_matrix, ckd_coeff_matrix, False, 0, 0)
        ckd_rand = ckd_prob > np.random.rand(len(self.population.index))
        old_ckd = self.population['ckd']
        new_ckd = ckd_rand & (in_care | out_care) & ~old_ckd

        # Save incidence
        denominator = (self.population.loc[in_care | out_care].groupby(['age_cat']).size()
                       .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='N'))['N']
        ckd_incidence = (self.population.loc[new_ckd].groupby(['age_cat']).size()
                         .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        ckd_incidence['N'] = denominator
        self.stats.ckd_incidence = self.stats.ckd_incidence.append(ckd_incidence)

        # Set variables
        self.population['ckd'] = (old_ckd | new_ckd).astype(int)

        # lipid
        lipid_coeff_matrix = self.parameters.lipid_coeff.to_numpy()
        lipid_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='lipid')
        lipid_prob = calculate_prob(lipid_pop_matrix, lipid_coeff_matrix, False, 0, 0)
        lipid_rand = lipid_prob > np.random.rand(len(self.population.index))
        old_lipid = self.population['lipid']
        new_lipid = lipid_rand & (in_care | out_care) & ~old_lipid

        # Save incidence
        lipid_incidence = (self.population.loc[new_lipid].groupby(['age_cat']).size()
                         .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        lipid_incidence['N'] = denominator
        self.stats.lipid_incidence = self.stats.lipid_incidence.append(lipid_incidence)

        # Set variables
        self.population['lipid'] = (old_lipid | new_lipid).astype(int)

        # diabetes
        diabetes_coeff_matrix = self.parameters.diabetes_coeff.to_numpy()
        diabetes_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='diabetes')
        diabetes_prob = calculate_prob(diabetes_pop_matrix, diabetes_coeff_matrix, False, 0, 0)
        diabetes_rand = diabetes_prob > np.random.rand(len(self.population.index))
        old_diabetes = self.population['diabetes']
        new_diabetes = diabetes_rand & (in_care | out_care) & ~old_diabetes

        # Save incidence
        diabetes_incidence = (self.population.loc[new_diabetes].groupby(['age_cat']).size()
                           .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                           .assign(year=self.year, replication=self.replication, group=self.group_name))
        diabetes_incidence['N'] = denominator
        self.stats.diabetes_incidence = self.stats.diabetes_incidence.append(diabetes_incidence)

        # Set variables
        self.population['diabetes'] = (old_diabetes | new_diabetes).astype(int)

        # hypertension
        hypertension_coeff_matrix = self.parameters.hypertension_coeff.to_numpy()
        hypertension_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='hypertension')
        hypertension_prob = calculate_prob(hypertension_pop_matrix, hypertension_coeff_matrix, False, 0, 0)
        hypertension_rand = hypertension_prob > np.random.rand(len(self.population.index))
        old_hypertension = self.population['hypertension']
        new_hypertension = hypertension_rand & (in_care | out_care) & ~old_hypertension

        # Save incidence
        hypertension_incidence = (self.population.loc[new_hypertension].groupby(['age_cat']).size()
                           .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                           .assign(year=self.year, replication=self.replication, group=self.group_name))
        hypertension_incidence['N'] = denominator
        self.stats.hypertension_incidence = self.stats.hypertension_incidence.append(hypertension_incidence)

        # Set variables
        self.population['hypertension'] = (old_hypertension | new_hypertension).astype(int)

    def record_stats(self):
        uninitiated = self.population['status'].isin([UNINITIATED_USER, UNINITIATED_NONUSER])
        in_care = self.population['status'] == ART_USER
        out_care = self.population['status'] == ART_NONUSER
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU
        alive = self.population['status'].isin([ART_USER, ART_NONUSER, REENGAGED, LTFU])

        # Count of new initiators by year
        if self.year == 2009:
            # Count of new initiators by year
            self.stats.new_init_count = (
                self.population.loc[uninitiated].groupby(['h1yy_orig']).size().reset_index(name='n').
                assign(replication=self.replication, group=self.group_name))

            # Count of new initiators by year and age
            self.stats.new_init_age = (
                self.population.loc[uninitiated].groupby(['h1yy_orig', 'age']).size().reset_index(name='n').
                assign(replication=self.replication, group=self.group_name))

        # Count of those in care by age_cat and year
        in_care_count = (self.population.loc[in_care | ltfu].groupby(['age_cat']).size()
                         .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.in_care_count = self.stats.in_care_count.append(in_care_count)

        # Count of those out of care by age_cat and year
        out_care_count = (self.population.loc[out_care | reengaged].groupby(['age_cat']).size()
                          .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                          .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.out_care_count = self.stats.out_care_count.append(out_care_count)

        # Count of those reengaging in care by age_cat and year
        reengaged_count = (self.population.loc[reengaged].groupby(['age_cat']).size()
                           .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                           .assign(year=(self.year + 1), replication=self.replication, group=self.group_name))
        self.stats.reengaged_count = self.stats.reengaged_count.append(reengaged_count)

        # Count of those lost to care by age_cat and year
        ltfu_count = (self.population.loc[ltfu].groupby(['age_cat']).size()
                      .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                      .assign(year=(self.year + 1), replication=self.replication, group=self.group_name))
        self.stats.ltfu_count = self.stats.ltfu_count.append(ltfu_count)

        # Count of those in care by age and year
        in_care_age = (self.population.loc[in_care | ltfu].groupby(['age']).size().reset_index(name='n')
                       .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.in_care_age = self.stats.in_care_age.append(in_care_age)

        # Count of those in care by age and year
        out_care_age = (self.population.loc[out_care | reengaged].groupby(['age']).size().reset_index(name='n')
                        .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.out_care_age = self.stats.out_care_age.append(out_care_age)

        # Count of those reengaging in care by age and year
        reengaged_age = (self.population.loc[reengaged].groupby(['age']).size().reset_index(name='n')
                         .assign(year=(self.year + 1), replication=self.replication, group=self.group_name))
        self.stats.reengaged_age = self.stats.reengaged_age.append(reengaged_age)

        # Count of those lost to care by age and year
        ltfu_age = (self.population.loc[ltfu].groupby(['age']).size().reset_index(name='n')
                    .assign(year=(self.year + 1), replication=self.replication, group=self.group_name))
        self.stats.ltfu_age = self.stats.ltfu_age.append(ltfu_age)

        # Keep track of unique individuals lost to follow up 2010-2015
        if 2010 <= self.year <= 2015:
            self.stats.unique_out_care_ids.update(self.population.loc[out_care | reengaged].index)


        if self.parameters.comorbidity_flag:
            # Encode set of comorbidities as an 8 bit integer
            multimorbidity_in_care = create_multimorbidity_stats(self.population.loc[in_care].copy())
            multimorbidity_in_care = multimorbidity_in_care.assign(year=self.year, replication=self.replication, group=self.group_name)
            self.stats.multimorbidity_in_care = self.stats.multimorbidity_in_care.append(multimorbidity_in_care)

    def record_final_stats(self):
        dead_in_care = self.population['status'] == DEAD_ART_USER
        dead_out_care = self.population['status'] == DEAD_ART_NONUSER

        # Count how many times people left and tally them up
        self.stats.n_times_lost = (pd.DataFrame(self.population['n_lost'].value_counts()).reset_index()
                                   .rename(columns={'n_lost': 'n', 'index': 'n_times_lost'}))
        self.stats.n_times_lost = (self.stats.n_times_lost.assign(
            pct=100.0 * self.stats.n_times_lost['n'] / self.stats.n_times_lost['n'].sum())
                                   .assign(replication=self.replication, group=self.group_name))

        # Count of those that died in care by age_cat and year
        self.stats.dead_in_care_count = (
            output_reindex(self.population.loc[dead_in_care].groupby(['year_died', 'age_cat']).size())
            .reset_index(name='n').rename(columns={'year_died': 'year'})
            .assign(replication=self.replication, group=self.group_name))


        # Count of those that died in care by age_cat and year
        self.stats.dead_out_care_count = (
            output_reindex(self.population.loc[dead_out_care].groupby(['year_died', 'age_cat']).size())
            .reset_index(name='n').rename(columns={'year_died': 'year'})
            .assign(replication=self.replication, group=self.group_name))

        # Count of those that died in care by age and year
        self.stats.dead_in_care_age = (
            self.population.loc[dead_in_care].groupby(['year_died', 'age']).size().reset_index(name='n')
            .rename(columns={'year_died': 'year'}).assign(replication=self.replication, group=self.group_name))

        # Count of those that died out of care by age and year
        self.stats.dead_out_care_age = (
            self.population.loc[dead_out_care].groupby(['year_died', 'age']).size().reset_index(name='n')
            .rename(columns={'year_died': 'year'}).assign(replication=self.replication, group=self.group_name))

        # Count how many years spent out of care and tally
        self.stats.years_out = (pd.DataFrame(self.population['years_out'].value_counts()).reset_index()
                                .rename(columns={'years_out': 'n', 'index': 'years_out'})
                                .assign(replication=self.replication, group=self.group_name))

        # Number of unique people out of care 2010-2015
        n_unique_out_care = (pd.DataFrame({'count': [len(self.stats.unique_out_care_ids)]})
                             .assign(replication=self.replication, group = self.group_name))

        # Record randomly generated numbers for sensitivity analysis
        random_params = pd.DataFrame({'mortality_in_care': self.parameters.mortality_in_care_rand,
                                      'mortality_out_care': self.parameters.mortality_out_care_rand,
                                      'loss_to_follow_up': self.parameters.loss_to_follow_up_rand,
                                      'cd4_increase': self.parameters.cd4_increase_rand,
                                      'cd4_decrease': self.parameters.cd4_decrease_rand,
                                      'lambda1_2009': self.parameters.age_in_2009_rand[0],
                                      'mu1_2009': self.parameters.age_in_2009_rand[1],
                                      'mu2_2009': self.parameters.age_in_2009_rand[2],
                                      'sigma1_2009': self.parameters.age_in_2009_rand[3],
                                      'sigma2_2009': self.parameters.age_in_2009_rand[4]}, index=[0]).assign(
            replication=self.replication, group=self.group_name)

        if self.parameters.comorbidity_flag:
            # Encode set of comorbidities as an 8 bit integer
            multimorbidity_dead = create_multimorbidity_stats(self.population.loc[dead_in_care | dead_out_care].copy())
            multimorbidity_dead = multimorbidity_dead.assign(year=self.year, replication=self.replication, group=self.group_name)
            self.stats.multimorbidity_dead = self.stats.multimorbidity_dead.append(multimorbidity_dead)

        # Make output directory if it doesn't exist
        os.makedirs(self.out_dir, exist_ok=True)

        # Save it all
        with pd.HDFStore(f'{self.out_dir}/{self.group_name}_{str(self.replication)}.h5') as store:
            store['in_care_count'] = self.stats.in_care_count
            store['in_care_age'] = self.stats.in_care_age
            store['out_care_count'] = self.stats.out_care_count
            store['out_care_age'] = self.stats.out_care_age
            store['reengaged_count'] = self.stats.reengaged_count
            store['reengaged_age'] = self.stats.reengaged_age
            store['ltfu_count'] = self.stats.ltfu_count
            store['ltfu_age'] = self.stats.ltfu_age
            store['dead_in_care_count'] = self.stats.dead_in_care_count
            store['dead_in_care_age'] = self.stats.dead_in_care_age
            store['dead_out_care_count'] = self.stats.dead_out_care_count
            store['dead_out_care_age'] = self.stats.dead_out_care_age
            store['new_init_count'] = self.stats.new_init_count
            store['new_init_age'] = self.stats.new_init_age
            store['years_out'] = self.stats.years_out
            store['n_times_lost'] = self.stats.n_times_lost
            store['n_unique_out_care'] = n_unique_out_care
            store['random_params'] = random_params

            if self.parameters.comorbidity_flag:
                store['multimorbidity_in_care'] = self.stats.multimorbidity_in_care
                store['multimorbidity_dead'] = self.stats.multimorbidity_dead

                store['anxiety_incidence'] = self.stats.anxiety_incidence
                store['depression_incidence'] = self.stats.depression_incidence
                store['ckd_incidence'] = self.stats.ckd_incidence
                store['lipid_incidence'] = self.stats.lipid_incidence
                store['diabetes_incidence'] = self.stats.diabetes_incidence
                store['hypertension_incidence'] = self.stats.hypertension_incidence


    def run(self):
        """ Simulate from 2010 to 2030 """
        while self.year <= 2030:

            # Everybody ages
            self.increment_age()

            # Apply comorbidities
            if self.parameters.comorbidity_flag:
                self.apply_stage_1()
                self.apply_stage_2()

            # In care operations
            self.increase_cd4_count()  # Increase cd4n in people in care
            self.add_new_dx()  # Add in newly diagnosed ART initiators
            self.kill_in_care()  # Kill some people in care
            self.lose_to_follow_up()  # Lose some people to follow up

            # Out of care operations
            self.decrease_cd4_count()  # Decrease cd4n in people out of care
            self.kill_out_care()  # Kill some people out of care
            self.reengage()  # Reengage some people out of care

            # Record output statistics
            self.record_stats()

            # Append changed populations to their respective DataFrames
            self.append_new()

            if self.verbose:
                print(self)

            # Increment year
            self.year += 1
        self.record_final_stats()

