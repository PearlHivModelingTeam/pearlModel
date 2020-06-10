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
DYING_ART_USER = 6
DYING_ART_NONUSER = 7
DEAD_ART_USER = 8
DEAD_ART_NONUSER = 9

# Smearing correction
SMEARING = 1.4


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
    init_cd4_cat = np.select([pop['last_init_sqrtcd4n'].lt(np.sqrt(200.0)),
                              pop['last_init_sqrtcd4n'].ge(np.sqrt(200.0)) & pop['last_init_sqrtcd4n'].lt(np.sqrt(350.0)),
                              pop['last_init_sqrtcd4n'].ge(np.sqrt(350.0)) & pop['last_init_sqrtcd4n'].lt(np.sqrt(500.0)),
                              pop['last_init_sqrtcd4n'].ge(np.sqrt(500.0))],
                             [1, 2, 3, 4])
    pop['cd4_cat_349'] = (init_cd4_cat == 2).astype(int)
    pop['cd4_cat_499'] = (init_cd4_cat == 3).astype(int)
    pop['cd4_cat_500'] = (init_cd4_cat == 4).astype(int)

    return pop


def calculate_cd4_increase(pop, knots, year, coeffs, vcov, sa):
    """ Calculate in care cd4 count via a linear function of time since art initiation, initial cd4 count, age
        category and cross terms"""

    # Calculate spline variables
    pop['time_from_h1yy'] = year - pop['last_h1yy']
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

    if sa is not None:
        # Calculate variance of prediction using matrix multiplication
        se = np.sqrt(np.sum(np.matmul(pop_matrix, vcov) * pop_matrix, axis=1))
        low = new_cd4 - 1.96 * se
        high = new_cd4 + 1.96 * se
        new_cd4 = (sa * (high - low)) + low

    return new_cd4

def calculate_cd4_decrease(pop, coeffs, sa, vcov):
    pop['time_out'] = pop['year'] - pop['ltfu_year']
    pop_matrix = pop[['intercept', 'time_out', 'sqrtcd4n_exit']].to_numpy()
    diff = np.matmul(pop_matrix, coeffs)

    if sa:
        se = np.sqrt(np.sum(np.matmul(pop_matrix, vcov) * pop_matrix, axis=1))
        low = diff - 1.96 * se
        high = diff + 1.96 * se
        diff = (sa * (high - low)) + low

    new_cd4 = np.sqrt((pop['sqrtcd4n_exit'].to_numpy() ** 2)*np.exp(diff) * SMEARING )
    return new_cd4

def create_comorbidity_pop_matrix(pop, condition):
    pop['time_since_art'] = pop['year'] - pop['h1yy']
    pop['out_care'] = (pop['status'] == ART_NONUSER).astype(int)

    if condition=='anxiety':
        return pop[['age', 'init_sqrtcd4n', 'depression', 'time_since_art', 'hcv', 'intercept', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='depression':
        return pop[['age', 'anxiety', 'init_sqrtcd4n', 'time_since_art', 'hcv', 'intercept', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='ckd':
        return pop[['age', 'anxiety', 'init_sqrtcd4n', 'diabetes', 'depression', 'time_since_art', 'hcv', 'hypertension', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='lipid':
        return pop[['age', 'anxiety', 'init_sqrtcd4n', 'ckd', 'diabetes', 'depression', 'time_since_art', 'hcv', 'hypertension', 'intercept', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='diabetes':
        return pop[['age', 'anxiety', 'init_sqrtcd4n', 'ckd', 'depression', 'time_since_art', 'hcv', 'hypertension', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition=='hypertension':
        return pop[['age', 'anxiety', 'init_sqrtcd4n', 'ckd', 'diabetes', 'depression', 'time_since_art', 'hcv', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy()

    elif condition in ['malig', 'esld', 'mi']:
        return pop[['age', 'anxiety', 'init_sqrtcd4n', 'ckd', 'diabetes', 'depression', 'time_since_art', 'hcv', 'hypertension', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy()

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

def calculate_prob(pop, coeffs, sa, vcov):
    """ Calculate the individual probability from logistic regression """

    log_odds = np.matmul(pop, coeffs)
    if sa is not None:
        # Calculate variance of prediction using matrix multiplication
        se = np.sqrt(np.sum(np.matmul(pop, vcov) * pop, axis=1))
        low = log_odds - 1.96 * se
        high = log_odds + 1.96 * se
        log_odds = (sa * (high - low)) + low

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

    # Compile list of number of new agents to be introduced in the model
    new_agents = new_dx.loc[np.arange(2010, 2031), ['art_users', 'art_nonusers']]
    new_agents['total'] = new_agents['art_users'] + new_agents['art_nonusers']

    return n_initial_nonusers, new_agents

def make_pop_2009(parameters, n_initial_nonusers, group_name):
    """ Create initial 2009 population. Draw ages from a mixed normal distribution truncated at 18 and 85. h1yy is
    assigned using proportions from NA-ACCORD data. Finally, sqrt cd4n is drawn from a 0-truncated normal for each
    h1yy """

    # Draw ages from the truncated mixed gaussian
    pop_size = parameters.on_art_2009[0].astype('int') + n_initial_nonusers
    population = simulate_ages(parameters.age_in_2009, pop_size)

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
    population['last_h1yy'] = population['h1yy']
    population['last_init_sqrtcd4n'] = population['init_sqrtcd4n']
    population = set_cd4_cat(population)
    population['init_age'] = population['age'] - (2009 - population['h1yy'])


    # Add final columns used for calculations and output
    population['n_lost'] = 0
    population['years_out'] = 0
    population['year_died'] = np.nan
    population['sqrtcd4n_exit'] = 0
    population['ltfu_year'] = 0
    population['return_year'] = 0
    population['intercept'] = 1.0
    population['year'] = 2009

    population['time_varying_sqrtcd4n'] = calculate_cd4_increase(population.copy(), parameters.cd4_increase_knots, 2009,
                                                                 parameters.cd4_increase.to_numpy(),
                                                                 parameters.cd4_increase_vcov.to_numpy(),
                                                                 parameters.cd4_increase_sa)
    # Set status and initiate out of care variables
    population['status'] = ART_USER
    non_user = np.random.choice(a=len(population.index), size=n_initial_nonusers, replace=False)
    years_out_of_care = np.random.choice(a=parameters.years_out_of_care['years'], size=n_initial_nonusers, p=parameters.years_out_of_care['probability'])
    population.loc[non_user, 'status'] = ART_NONUSER
    population.loc[non_user, 'sqrtcd4n_exit'] = population.loc[n_initial_nonusers, 'time_varying_sqrtcd4n']
    population.loc[non_user, 'ltfu_year'] = 2009
    population.loc[non_user, 'return_year'] = 2009 + years_out_of_care
    population.loc[non_user, 'n_lost'] += 1

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

        # Stage 3 comorbidities
        population['malig'] = (np.random.rand(len(population.index)) < parameters.malig_prev_users.values).astype(int)
        population['esld'] = (np.random.rand(len(population.index)) < parameters.esld_prev_users.values).astype(int)
        population['mi'] = (np.random.rand(len(population.index)) < parameters.mi_prev_users.values).astype(int)

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population

def make_new_population(parameters, n_new_agents, pop_size_2009, group_name, replication, stats):
    """ Draw ages for new art initiators """

    # Draw a random value between predicted and 2018 predicted value for years greater than 2018
    rand = np.random.rand(len(parameters.age_by_h1yy.index))
    parameters.age_by_h1yy['estimate'] = rand * (parameters.age_by_h1yy['high_value'] - parameters.age_by_h1yy['low_value']) + parameters.age_by_h1yy['low_value']

    stats.art_coeffs = parameters.age_by_h1yy[['estimate']].assign(group=group_name, replication=replication).reset_index()

    rand = np.random.rand(len(parameters.cd4n_by_h1yy.index))
    parameters.cd4n_by_h1yy['estimate'] = rand * (parameters.cd4n_by_h1yy['high_value'] - parameters.cd4n_by_h1yy['low_value']) + parameters.cd4n_by_h1yy['low_value']

    # Create population
    population = pd.DataFrame()
    for h1yy in parameters.age_by_h1yy.index.levels[0]:
        n_users = n_new_agents.loc[h1yy, 'art_users']
        n_nonusers = n_new_agents.loc[h1yy, 'art_nonusers']
        grouped_pop = simulate_ages(parameters.age_by_h1yy.loc[h1yy], n_users + n_nonusers)
        grouped_pop['h1yy'] = h1yy
        grouped_pop['status'] = UNINITIATED_USER
        non_users = np.random.choice(a=len(grouped_pop.index), size=n_nonusers, replace=False)
        grouped_pop.loc[non_users, 'status'] = UNINITIATED_NONUSER

        population = pd.concat([population, grouped_pop])

    population['age'] = np.floor(population['age'])
    population['age_cat'] = np.floor(population['age'] / 10)
    population.loc[population['age_cat'] < 2, 'age_cat'] = 2
    population.loc[population['age_cat'] > 7, 'age_cat'] = 7

    # Add id number
    population['id'] = np.arange(pop_size_2009, (pop_size_2009 + population.index.size))

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using
    population = population.set_index('h1yy')
    for h1yy, group in population.groupby(level=0):
        mu = parameters.cd4n_by_h1yy.loc[(h1yy, 'mu'), 'estimate']
        sigma = parameters.cd4n_by_h1yy.loc[(h1yy, 'sigma'), 'estimate']
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size)
        population.loc[h1yy, 'time_varying_sqrtcd4n'] = sqrt_cd4n

    population = population.reset_index().set_index('id').sort_index()

    # Calculate time varying cd4 count
    population['last_h1yy'] = population['h1yy']
    population['init_sqrtcd4n'] = population['time_varying_sqrtcd4n']
    population['last_init_sqrtcd4n'] = population['init_sqrtcd4n']
    population['init_age'] = population['age']
    population = set_cd4_cat(population)

    # Add final columns used for calculations and output
    population['n_lost'] = 0
    population['years_out'] = 0
    population['year_died'] = np.nan
    population['sqrtcd4n_exit'] = 0
    population['ltfu_year'] = 0
    population['return_year'] = 0
    population['intercept'] = 1.0
    population['year'] = 2009

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

        # Stage 3 comorbidities
        population['malig'] = (np.random.rand(len(population.index)) < parameters.malig_prev_inits.values).astype(int)
        population['esld'] = (np.random.rand(len(population.index)) < parameters.esld_prev_inits.values).astype(int)
        population['mi'] = (np.random.rand(len(population.index)) < parameters.mi_prev_inits.values).astype(int)

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population

def create_multimorbidity_stats(pop):
    # Encode multimorbidity as 8 bit integer
    df = pop[['age_cat', 'smoking', 'hcv', 'anxiety', 'depression', 'ckd', 'lipid', 'diabetes', 'hypertension', 'malig', 'esld', 'mi']].copy()
    df['multimorbidity'] = (
            df['smoking'].map(str) + df['hcv'].map(str) + df['anxiety'].map(str) + df['depression'].map(str)
            + df['ckd'].map(str) + df['lipid'].map(str) + df['diabetes'].map(str) + df['hypertension'].map(
        str)).apply(int, base=2)

    # Count how many people have each unique set of comorbidities
    df = df.groupby(['age_cat', 'multimorbidity']).size()
    index = pd.MultiIndex.from_product([df.index.levels[0], range(2048)], names=['age_cat', 'multimorbidity'])
    df = df.reindex(index=index, fill_value=0).reset_index(name='n')

    return (df)


###############################################################################
# Parameter and Statistics Classes                                            #
###############################################################################

class Parameters:
    def __init__(self, path, group_name, comorbidity_flag, sa_dict, new_dx='base', output_folder=f'{os.getcwd()}/../../out/raw', record_tv_cd4=False):
        self.output_folder = output_folder
        self.record_tv_cd4 = record_tv_cd4
        # Unpack Sensitivity Analysis List
        lambda1_sa = sa_dict['lambda1']
        mu1_sa = sa_dict['mu1']
        mu2_sa = sa_dict['mu2']
        sigma1_sa = sa_dict['sigma1']
        sigma2_sa = sa_dict['sigma2']

        # 2009 population
        self.on_art_2009 = pd.read_hdf(path, 'on_art_2009').loc[group_name]
        self.age_in_2009 = pd.read_hdf(path, 'age_in_2009').loc[group_name]
        self.h1yy_by_age_2009 = pd.read_hdf(path, 'h1yy_by_age_2009')
        self.cd4n_by_h1yy_2009 = pd.read_hdf(path, 'cd4n_by_h1yy_2009').loc[group_name]

        # Age in 2009 sensitivity analysis
        if lambda1_sa==0:
            self.age_in_2009.loc['lambda1', 'estimate'] = self.age_in_2009.loc['lambda1', 'conf_low']
        elif lambda1_sa==1:
            self.age_in_2009.loc['lambda1', 'estimate'] = self.age_in_2009.loc['lambda1', 'conf_high']

        if mu1_sa==0:
            self.age_in_2009.loc['mu1', 'estimate'] = self.age_in_2009.loc['mu1', 'conf_low']
        elif mu1_sa==1:
            self.age_in_2009.loc['mu1', 'estimate'] = self.age_in_2009.loc['mu1', 'conf_high']

        if mu2_sa==0:
            self.age_in_2009.loc['mu2', 'estimate'] = self.age_in_2009.loc['mu2', 'conf_low']
        elif mu2_sa==1:
            self.age_in_2009.loc['mu2', 'estimate'] = self.age_in_2009.loc['mu2', 'conf_high']

        if sigma1_sa==0:
            self.age_in_2009.loc['sigma1', 'estimate'] = self.age_in_2009.loc['sigma1', 'conf_low']
        elif sigma1_sa==1:
            self.age_in_2009.loc['sigma1', 'estimate'] = self.age_in_2009.loc['sigma1', 'conf_high']

        if sigma2_sa==0:
            self.age_in_2009.loc['sigma2', 'estimate'] = self.age_in_2009.loc['sigma2', 'conf_low']
        elif sigma2_sa==1:
            self.age_in_2009.loc['sigma2', 'estimate'] = self.age_in_2009.loc['sigma2', 'conf_high']

        # New ART initiators
        if new_dx == 'base':
            self.new_dx = pd.read_hdf(path, 'new_dx').loc[group_name]
        elif new_dx == 'ehe':
            self.new_dx = pd.read_hdf(path, 'new_dx_ehe').loc[group_name]
        elif new_dx == 'sa':
            self.new_dx = pd.read_hdf(path, 'new_dx_sa').loc[group_name]
        else:
            raise ValueError('Invalid new diagnosis file specified')


        # Sensitivity analysis for new diagnoses
        if sa_dict['new_pop_size'] == 0:
            self.new_dx['upper'] = self.new_dx['lower']
        elif sa_dict['new_pop_size'] ==1:
            self.new_dx['lower'] = self.new_dx['upper']

        self.linkage_to_care = pd.read_hdf(path, 'linkage_to_care').loc[group_name]
        self.age_by_h1yy = pd.read_hdf(path, 'age_by_h1yy').loc[group_name]
        self.cd4n_by_h1yy = pd.read_hdf(path, 'cd4n_by_h1yy').loc[group_name]

        # Mortality In Care
        self.mortality_in_care = pd.read_hdf(path, 'mortality_in_care').loc[group_name]
        self.mortality_in_care_vcov = pd.read_hdf(path, 'mortality_in_care_vcov').loc[group_name]
        self.mortality_in_care_sa = sa_dict['mortality_in_care']

        # Mortality Out Of Care
        self.mortality_out_care = pd.read_hdf(path, 'mortality_out_care').loc[group_name]
        self.mortality_out_care_vcov = pd.read_hdf(path, 'mortality_out_care_vcov').loc[group_name]
        self.mortality_out_care_sa = sa_dict['mortality_out_care']

        # Loss To Follow Up
        self.loss_to_follow_up = pd.read_hdf(path, 'loss_to_follow_up').loc[group_name]
        self.loss_to_follow_up_vcov = pd.read_hdf(path, 'loss_to_follow_up_vcov').loc[group_name]
        self.loss_to_follow_up_sa = sa_dict['loss_to_follow_up']
        self.ltfu_knots = pd.read_hdf(path, 'ltfu_knots').loc[group_name]

        # Cd4 Increase
        self.cd4_increase = pd.read_hdf(path, 'cd4_increase').loc[group_name]
        self.cd4_increase_vcov = pd.read_hdf(path, 'cd4_increase_vcov').loc[group_name]
        self.cd4_increase_sa = sa_dict['cd4_increase']
        self.cd4_increase_knots = pd.read_hdf(path, 'cd4_increase_knots').loc[group_name]

        # Cd4 Decrease
        self.cd4_decrease = pd.read_hdf(path, 'cd4_decrease').loc['all']
        self.cd4_decrease_vcov = pd.read_hdf(path, 'cd4_decrease_vcov')
        self.cd4_decrease_sa = sa_dict['cd4_decrease']

        # Years out of Care
        self.years_out_of_care = pd.read_hdf(path, 'years_out_of_care')

        # Stage 0 Comorbidities
        self.hcv_prev_users = pd.read_hdf(path, 'hcv_prev_users').loc[group_name]
        self.smoking_prev_users = pd.read_hdf(path, 'smoking_prev_users').loc[group_name]

        self.hcv_prev_inits = pd.read_hdf(path, 'hcv_prev_inits').loc[group_name]
        self.smoking_prev_inits = pd.read_hdf(path, 'smoking_prev_inits').loc[group_name]

        # Stage 1 Comorbidities
        self.anxiety_prev_users = pd.read_hdf(path, 'anxiety_prev_users').loc[group_name]
        self.depression_prev_users = pd.read_hdf(path, 'depression_prev_users').loc[group_name]
        self.anxiety_prev_inits = pd.read_hdf(path, 'anxiety_prev_inits').loc[group_name]
        self.depression_prev_inits = pd.read_hdf(path, 'depression_prev_inits').loc[group_name]
        self.anxiety_coeff = pd.read_hdf(path, 'anxiety_coeff').loc[group_name]
        self.depression_coeff = pd.read_hdf(path, 'depression_coeff').loc[group_name]

        # Stage 2 Comorbidities
        self.ckd_prev_users = pd.read_hdf(path, 'ckd_prev_users').loc[group_name]
        self.lipid_prev_users = pd.read_hdf(path, 'lipid_prev_users').loc[group_name]
        self.diabetes_prev_users = pd.read_hdf(path, 'diabetes_prev_users').loc[group_name]
        self.hypertension_prev_users = pd.read_hdf(path, 'hypertension_prev_users').loc[group_name]
        self.ckd_prev_inits = pd.read_hdf(path, 'ckd_prev_inits').loc[group_name]
        self.lipid_prev_inits = pd.read_hdf(path, 'lipid_prev_inits').loc[group_name]
        self.diabetes_prev_inits = pd.read_hdf(path, 'diabetes_prev_inits').loc[group_name]
        self.hypertension_prev_inits = pd.read_hdf(path, 'hypertension_prev_inits').loc[group_name]
        self.ckd_coeff = pd.read_hdf(path, 'ckd_coeff').loc[group_name]
        self.lipid_coeff = pd.read_hdf(path, 'lipid_coeff').loc[group_name]
        self.diabetes_coeff = pd.read_hdf(path, 'diabetes_coeff').loc[group_name]
        self.hypertension_coeff = pd.read_hdf(path, 'hypertension_coeff').loc[group_name]

        # Stage 3 Comorbidities
        self.malig_prev_users = pd.read_hdf(path, 'malig_prev_users').loc[group_name]
        self.esld_prev_users = pd.read_hdf(path, 'esld_prev_users').loc[group_name]
        self.mi_prev_users = pd.read_hdf(path, 'mi_prev_users').loc[group_name]
        self.malig_prev_inits = pd.read_hdf(path, 'malig_prev_inits').loc[group_name]
        self.esld_prev_inits = pd.read_hdf(path, 'esld_prev_inits').loc[group_name]
        self.mi_prev_inits = pd.read_hdf(path, 'mi_prev_inits').loc[group_name]
        self.malig_coeff = pd.read_hdf(path, 'malig_coeff').loc[group_name]
        self.esld_coeff = pd.read_hdf(path, 'esld_coeff').loc[group_name]
        self.mi_coeff = pd.read_hdf(path, 'mi_coeff').loc[group_name]

        # Comortality
        self.comorbidity_flag = comorbidity_flag
        if self.comorbidity_flag:
            self.mortality_in_care = pd.read_hdf(path, 'mortality_in_care_co').loc[group_name]
            self.mortality_out_care = pd.read_hdf(path, 'mortality_out_care_co').loc[group_name]



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
        self.art_coeffs = pd.DataFrame()
        self.median_cd4s = pd.DataFrame()
        self.tv_cd4_2009 = pd.DataFrame()

        # Multimorbidity
        self.multimorbidity_in_care = pd.DataFrame()
        self.multimorbidity_inits = pd.DataFrame()
        self.multimorbidity_dead = pd.DataFrame()

        # Incidence
        self.anxiety_incidence = pd.DataFrame()
        self.depression_incidence = pd.DataFrame()
        self.ckd_incidence = pd.DataFrame()
        self.lipid_incidence = pd.DataFrame()
        self.diabetes_incidence = pd.DataFrame()
        self.hypertension_incidence = pd.DataFrame()
        self.malig_incidence = pd.DataFrame()
        self.esld_incidence = pd.DataFrame()
        self.mi_incidence = pd.DataFrame()


def output_reindex(df):
    """ Helper function for reindexing output tables """
    return df.reindex(pd.MultiIndex.from_product([df.index.levels[0], np.arange(2.0, 8.0)], names=['year', 'age_cat']),
                      fill_value=0)


###############################################################################
# Pearl Class                                                                 #
###############################################################################

class Pearl:
    def __init__(self, parameters, group_name, replication, verbose=False):
        self.group_name = group_name
        self.replication = replication
        self.verbose = verbose
        self.year = 2009
        self.parameters = parameters

        # Initiate output class
        self.stats = Statistics()

        # Simulate number of new art initiators
        n_initial_nonusers, n_new_agents = simulate_new_dx(parameters.new_dx, parameters.linkage_to_care)

        # Create 2009 population
        self.population = make_pop_2009(parameters, n_initial_nonusers, self.group_name)

        # Create population of new art initiators
        self.population = self.population.append(
            make_new_population(parameters, n_new_agents, len(self.population.index), self.group_name, self.replication, self.stats))


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
        uninitiated_user = len(self.population.loc[self.population['status'].isin([UNINITIATED_USER])])
        uninitiated_nonuser = len(self.population.loc[self.population['status'].isin([UNINITIATED_NONUSER])])

        string = 'Year End: ' + str(self.year) + '\n'
        string += 'Total Population Size: ' + str(total) + '\n'
        string += 'In Care Size: ' + str(in_care) + '\n'
        string += 'Out Care Size: ' + str(out_care) + '\n'
        string += 'Dead In Care Size: ' + str(dead_in_care) + '\n'
        string += 'Dead Out Care Size: ' + str(dead_out_care) + '\n'
        string += 'Uninitiated User Size: ' + str(uninitiated_user) + '\n'
        string += 'Uninitiated Nonuser Size: ' + str(uninitiated_nonuser) + '\n'
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
            self.parameters.cd4_increase_sa)


    def decrease_cd4_count(self):
        out_care = self.population['status'] == ART_NONUSER
        self.population.loc[out_care, 'time_varying_sqrtcd4n'] = calculate_cd4_decrease(
            self.population.loc[out_care].copy(),
            self.parameters.cd4_decrease.to_numpy(),
            self.parameters.cd4_decrease_sa,
            self.parameters.cd4_decrease_vcov.to_numpy())


    def add_new_user(self):
        new_user = (self.population['status'] == UNINITIATED_USER) & (self.population['h1yy'] == self.year)
        self.population.loc[new_user, 'status'] = ART_USER

    def add_new_nonuser(self):
        new_nonuser = (self.population['status'] == UNINITIATED_NONUSER) & (self.population['h1yy'] == self.year)
        self.population.loc[new_nonuser, 'status'] = ART_NONUSER
        self.population.loc[new_nonuser, 'sqrtcd4n_exit'] = self.population.loc[new_nonuser, 'time_varying_sqrtcd4n']
        self.population.loc[new_nonuser, 'ltfu_year'] = self.year
        self.population.loc[new_nonuser, 'n_lost'] += 1

        # Allow New Nonusers to enter care
        n_new_nonusers = len(self.population.loc[new_nonuser])
        years_out_of_care = np.random.choice(a=self.parameters.years_out_of_care['years'], size=n_new_nonusers,
                                             p=self.parameters.years_out_of_care['probability'])
        self.population.loc[new_nonuser, 'return_year'] = self.year + years_out_of_care

    def kill_in_care(self):
        in_care = self.population['status'] == ART_USER
        coeff_matrix = self.parameters.mortality_in_care.to_numpy()
        if self.parameters.comorbidity_flag:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'init_sqrtcd4n', 'h1yy', 'smoking', 'hcv', 'anxiety', 'depression', 'hypertension', 'diabetes', 'ckd', 'lipid']].to_numpy()
        else:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'init_sqrtcd4n', 'h1yy']].to_numpy()
        vcov_matrix = self.parameters.mortality_in_care_vcov.to_numpy()
        death_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.mortality_in_care_sa,
                                    vcov_matrix)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & in_care
        self.population.loc[died, 'status'] = DYING_ART_USER
        self.population.loc[died, 'year_died'] = self.year

    def kill_out_care(self):
        out_care = self.population['status'] == ART_NONUSER
        coeff_matrix = self.parameters.mortality_out_care.to_numpy()
        if self.parameters.comorbidity_flag:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'time_varying_sqrtcd4n', 'smoking', 'hcv', 'anxiety', 'depression', 'hypertension', 'diabetes', 'ckd', 'lipid']].to_numpy()
        else:
            pop_matrix = self.population[['intercept', 'year', 'age_cat', 'time_varying_sqrtcd4n']].to_numpy()
        vcov_matrix = self.parameters.mortality_out_care_vcov.to_numpy()
        death_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.mortality_out_care_sa,
                                    vcov_matrix)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & out_care
        self.population.loc[died, 'status'] = DYING_ART_NONUSER
        self.population.loc[died, 'year_died'] = self.year
        self.population.loc[died, 'return_year'] = 0

    def lose_to_follow_up(self):
        in_care = self.population['status'] == ART_USER
        coeff_matrix = self.parameters.loss_to_follow_up.to_numpy()
        vcov_matrix = self.parameters.loss_to_follow_up_vcov.to_numpy()
        pop_matrix = create_ltfu_pop_matrix(self.population.copy(), self.parameters.ltfu_knots)
        ltfu_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.loss_to_follow_up_sa,
                                   vcov_matrix)
        lost = (ltfu_prob > np.random.rand(len(self.population.index))) & in_care
        n_lost = len(self.population.loc[lost])
        years_out_of_care = np.random.choice(a=self.parameters.years_out_of_care['years'], size=n_lost,
                                             p=self.parameters.years_out_of_care['probability'])
        self.population.loc[lost, 'return_year'] = self.year + years_out_of_care
        self.population.loc[lost, 'status'] = LTFU
        self.population.loc[lost, 'sqrtcd4n_exit'] = self.population.loc[lost, 'time_varying_sqrtcd4n']
        self.population.loc[lost, 'ltfu_year'] = self.year
        self.population.loc[lost, 'n_lost'] += 1

    def reengage(self):
        out_care = self.population['status'] == ART_NONUSER
        reengaged = (self.year == self.population['return_year']) & out_care

        self.population.loc[reengaged, 'status'] = REENGAGED

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        self.population.loc[reengaged, 'last_init_sqrtcd4n'] = self.population.loc[reengaged, 'time_varying_sqrtcd4n']
        self.population.loc[reengaged, 'last_h1yy'] = self.year
        self.population = set_cd4_cat(self.population)
        self.population.loc[reengaged, 'return_year'] = 0


        # Save years out of care
        years_out = (pd.DataFrame(self.population.loc[reengaged, 'years_out'].value_counts()).reindex(range(1, 8), fill_value=0).reset_index()
                     .rename(columns={'index': 'years', 'years_out': 'n'})
                     .assign(group= self.group_name, replication=self.replication, year=self.year))
        self.stats.years_out = self.stats.years_out.append(years_out)
        self.population.loc[reengaged, 'years_out'] = 0

    def append_new(self):
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU
        dying_art_user = self.population['status'] == DYING_ART_USER
        dying_art_nonuser = self.population['status'] == DYING_ART_NONUSER

        self.population.loc[reengaged, 'status'] = ART_USER
        self.population.loc[ltfu, 'status'] = ART_NONUSER
        self.population.loc[dying_art_user, 'status'] = DEAD_ART_USER
        self.population.loc[dying_art_nonuser, 'status'] = DEAD_ART_NONUSER

    def apply_stage_1(self):
        in_care = self.population['status'] == ART_USER
        out_care = self.population['status'] == ART_NONUSER

        # Use matrix multiplication to calculate probability of anxiety incidence
        anxiety_coeff_matrix = self.parameters.anxiety_coeff.to_numpy()
        anxiety_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition ='anxiety')
        anxiety_prob = calculate_prob(anxiety_pop_matrix, anxiety_coeff_matrix, None, None)
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
        depression_prob = calculate_prob(depression_pop_matrix, depression_coeff_matrix, None, None)
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
        ckd_prob = calculate_prob(ckd_pop_matrix, ckd_coeff_matrix, None, None)
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
        lipid_prob = calculate_prob(lipid_pop_matrix, lipid_coeff_matrix, None, None)
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
        diabetes_prob = calculate_prob(diabetes_pop_matrix, diabetes_coeff_matrix, None, None)
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
        hypertension_prob = calculate_prob(hypertension_pop_matrix, hypertension_coeff_matrix, None, None)
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

    def apply_stage_3(self):
        in_care = self.population['status'] == ART_USER
        out_care = self.population['status'] == ART_NONUSER

        # malig
        malig_coeff_matrix = self.parameters.malig_coeff.to_numpy()
        malig_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='malig')
        malig_prob = calculate_prob(malig_pop_matrix, malig_coeff_matrix, None, None)
        malig_rand = malig_prob > np.random.rand(len(self.population.index))
        old_malig = self.population['malig']
        new_malig = malig_rand & (in_care | out_care) & ~old_malig

        # Save incidence
        denominator = (self.population.loc[in_care | out_care].groupby(['age_cat']).size()
                       .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='N'))['N']
        malig_incidence = (self.population.loc[new_malig].groupby(['age_cat']).size()
                         .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        malig_incidence['N'] = denominator
        self.stats.malig_incidence = self.stats.malig_incidence.append(malig_incidence)

        # Set variables
        self.population['malig'] = (old_malig | new_malig).astype(int)

        # esld
        esld_coeff_matrix = self.parameters.esld_coeff.to_numpy()
        esld_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='esld')
        esld_prob = calculate_prob(esld_pop_matrix, esld_coeff_matrix, None, None)
        esld_rand = esld_prob > np.random.rand(len(self.population.index))
        old_esld = self.population['esld']
        new_esld = esld_rand & (in_care | out_care) & ~old_esld

        # Save incidence
        denominator = (self.population.loc[in_care | out_care].groupby(['age_cat']).size()
                       .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='N'))['N']
        esld_incidence = (self.population.loc[new_esld].groupby(['age_cat']).size()
                           .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                           .assign(year=self.year, replication=self.replication, group=self.group_name))
        esld_incidence['N'] = denominator
        self.stats.esld_incidence = self.stats.esld_incidence.append(esld_incidence)

        # Set variables
        self.population['esld'] = (old_esld | new_esld).astype(int)

        # mi
        mi_coeff_matrix = self.parameters.mi_coeff.to_numpy()
        mi_pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition='mi')
        mi_prob = calculate_prob(mi_pop_matrix, mi_coeff_matrix, None, None)
        mi_rand = mi_prob > np.random.rand(len(self.population.index))
        old_mi = self.population['mi']
        new_mi = mi_rand & (in_care | out_care) & ~old_mi

        # Save incidence
        denominator = (self.population.loc[in_care | out_care].groupby(['age_cat']).size()
                       .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='N'))['N']
        mi_incidence = (self.population.loc[new_mi].groupby(['age_cat']).size()
                          .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                          .assign(year=self.year, replication=self.replication, group=self.group_name))
        mi_incidence['N'] = denominator
        self.stats.mi_incidence = self.stats.mi_incidence.append(mi_incidence)

        # Set variables
        self.population['esld'] = (old_esld | new_esld).astype(int)

    def record_stats(self):
        stay_in_care = self.population['status'] == ART_USER
        stay_out_care = self.population['status'] == ART_NONUSER
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU
        dying_art_user = self.population['status'] == DYING_ART_USER
        dying_art_nonuser = self.population['status'] == DYING_ART_NONUSER
        in_care = stay_in_care | ltfu | dying_art_user
        out_care = stay_out_care | ltfu | dying_art_nonuser


        # Count of those in care by age_cat and year
        in_care_count = (self.population.loc[in_care].groupby(['age_cat']).size()
                         .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.in_care_count = self.stats.in_care_count.append(in_care_count)

        # Count of those out of care by age_cat and year
        out_care_count = (self.population.loc[out_care].groupby(['age_cat']).size()
                          .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                          .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.out_care_count = self.stats.out_care_count.append(out_care_count)

        # Count of those reengaging in care by age_cat and year
        reengaged_count = (self.population.loc[reengaged].groupby(['age_cat']).size()
                           .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                           .assign(year=(self.year), replication=self.replication, group=self.group_name))
        self.stats.reengaged_count = self.stats.reengaged_count.append(reengaged_count)

        # Count of those lost to care by age_cat and year
        ltfu_count = (self.population.loc[ltfu].groupby(['age_cat']).size()
                      .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                      .assign(year=(self.year), replication=self.replication, group=self.group_name))
        self.stats.ltfu_count = self.stats.ltfu_count.append(ltfu_count)

        # Count of those in care by age and year
        in_care_age = (self.population.loc[in_care].groupby(['age']).size().reset_index(name='n')
                       .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.in_care_age = self.stats.in_care_age.append(in_care_age)

        # Count of those in care by age and year
        out_care_age = (self.population.loc[out_care].groupby(['age']).size().reset_index(name='n')
                        .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.out_care_age = self.stats.out_care_age.append(out_care_age)

        # Count of those reengaging in care by age and year
        reengaged_age = (self.population.loc[reengaged].groupby(['age']).size().reset_index(name='n')
                         .assign(year=(self.year), replication=self.replication, group=self.group_name))
        self.stats.reengaged_age = self.stats.reengaged_age.append(reengaged_age)

        # Count of those lost to care by age and year
        ltfu_age = (self.population.loc[ltfu].groupby(['age']).size().reset_index(name='n')
                    .assign(year=(self.year), replication=self.replication, group=self.group_name))
        self.stats.ltfu_age = self.stats.ltfu_age.append(ltfu_age)

        # Keep track of unique individuals lost to follow up 2010-2015
        if 2010 <= self.year <= 2015:
            self.stats.unique_out_care_ids.update(self.population.loc[out_care].index)

        # Report median initial and varying cd4 count for in care population
        median_init_cd4 = self.population.loc[self.population['h1yy']==self.year, 'init_sqrtcd4n'].median()
        median_tv_cd4 = self.population.loc[in_care, 'time_varying_sqrtcd4n'].median()
        median_tv_cd4_out = self.population.loc[out_care, 'time_varying_sqrtcd4n'].median()
        median_cd4s = pd.DataFrame({'init_cd4': median_init_cd4,
                                    'tv_cd4': median_tv_cd4,
                                    'tv_cd4_out': median_tv_cd4_out,
                                    'year': self.year,
                                    'replication': self.replication,
                                    'group': self.group_name}, index=[0])
        self.stats.median_cd4s = pd.concat([self.stats.median_cd4s, median_cd4s])

        if (self.year==2009) & (self.parameters.record_tv_cd4):
            self.tv_cd4_2009 = pd.DataFrame(self.population.loc[in_care, 'time_varying_sqrtcd4n']).assign(group=self.group_name, replication=self.replication)

        # Encode set of comorbidities as an 8 bit integer
        if self.parameters.comorbidity_flag:
            multimorbidity_in_care = create_multimorbidity_stats(self.population.loc[in_care].copy())
            multimorbidity_in_care = multimorbidity_in_care.assign(year=self.year, replication=self.replication, group=self.group_name)
            self.stats.multimorbidity_in_care = self.stats.multimorbidity_in_care.append(multimorbidity_in_care)

            multimorbidity_inits = create_multimorbidity_stats(self.population.loc[self.population['h1yy'] == self.year].copy())
            multimorbidity_inits = multimorbidity_inits.assign(year=self.year, replication=self.replication, group=self.group_name)
            self.stats.multimorbidity_inits = self.stats.multimorbidity_inits.append(multimorbidity_inits)

    def record_final_stats(self):
        dead_in_care = self.population['status'] == DEAD_ART_USER
        dead_out_care = self.population['status'] == DEAD_ART_NONUSER

        # Count of new initiators by year
        self.stats.new_init_count = (
            self.population.groupby(['h1yy']).size().reset_index(name='n').
                assign(replication=self.replication, group=self.group_name))

        # Count of new initiators by year and age
        self.stats.new_init_age = (
            self.population.groupby(['h1yy', 'init_age']).size().reset_index(name='n').
                assign(replication=self.replication, group=self.group_name))

        # Record initial CD4 Counts
        initial_cd4n = self.population[['init_sqrtcd4n', 'h1yy']].copy()
        initial_cd4n['init_sqrtcd4n'] = initial_cd4n['init_sqrtcd4n'].astype(int)
        initial_cd4n = initial_cd4n.groupby(['h1yy', 'init_sqrtcd4n']).size()
        initial_cd4n = initial_cd4n.reindex(pd.MultiIndex.from_product([np.arange(2000, 2030), np.arange(51)], names=['h1yy', 'init_sqrtcd4n']), fill_value=0).reset_index(name='n')
        initial_cd4n = initial_cd4n.assign(group=self.group_name, replication=self.replication)

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

        # Number of unique people out of care 2010-2015
        n_unique_out_care = (pd.DataFrame({'count': [len(self.stats.unique_out_care_ids)]})
                             .assign(replication=self.replication, group = self.group_name))

        if self.parameters.comorbidity_flag:
            # Encode set of comorbidities as an 8 bit integer
            multimorbidity_dead = create_multimorbidity_stats(self.population.loc[dead_in_care | dead_out_care].copy())
            multimorbidity_dead = multimorbidity_dead.assign(year=self.year, replication=self.replication, group=self.group_name)
            self.stats.multimorbidity_dead = self.stats.multimorbidity_dead.append(multimorbidity_dead)

        # Make output directory if it doesn't exist
        os.makedirs(self.parameters.output_folder, exist_ok=True)

        # Save it all
        with pd.HDFStore(f'{self.parameters.output_folder}/{self.group_name}_{str(self.replication)}.h5') as store:
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
            store['initial_cd4n'] = initial_cd4n
            store['n_unique_out_care'] = n_unique_out_care
            store['art_coeffs'] = self.stats.art_coeffs
            store['median_cd4s'] = self.stats.median_cd4s
            if self.parameters.record_tv_cd4:
                store['tv_cd4_2009'] = self.tv_cd4_2009

            if self.parameters.comorbidity_flag:
                store['multimorbidity_in_care'] = self.stats.multimorbidity_in_care
                store['multimorbidity_inits'] = self.stats.multimorbidity_inits
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
                self.apply_stage_3()

            # In care operations
            self.increase_cd4_count()  # Increase cd4n in people in care
            self.add_new_user()  # Add in newly diagnosed ART initiators
            self.kill_in_care()  # Kill some people in care
            self.lose_to_follow_up()  # Lose some people to follow up

            # Out of care operations
            self.decrease_cd4_count()  # Decrease cd4n in people out of care
            self.add_new_nonuser()  # Add in newly diagnosed ART initiators
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

