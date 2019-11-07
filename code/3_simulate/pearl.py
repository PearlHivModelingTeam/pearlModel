# Imports
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import feather

pd.set_option("display.max_rows", 1001)

# Status Constants
UNINITIATED = 0
IN_CARE = 1
OUT_CARE = 2
REENGAGED = 3
LTFU = 4
DEAD_IN_CARE = 5
DEAD_OUT_CARE = 6


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
    a_mod = (a - mu) / sigma
    b_mod = (b - mu) / sigma
    return stats.truncnorm.rvs(a_mod, b_mod, loc=mu, scale=sigma, size=n)


def sim_pop(coeffs, pop_size):
    """ Draw ages from a mixed or single gaussian truncated at 18 and 85 given the coefficients and population size."""

    components = np.random.choice([1, 2], size=pop_size,
                                  p=[coeffs.loc['lambda1', 'estimate'], coeffs.loc['lambda2', 'estimate']],
                                  replace=True)
    pop_size_1 = (components == 1).sum()
    pop_size_2 = (components == 2).sum()

    # Draw age from each respective truncated normal
    if pop_size_1 == 0:
        population = draw_from_trunc_norm(18, 85, coeffs.loc['mu2', 'estimate'], coeffs.loc['sigma2', 'estimate'],
                                          pop_size_2)
    elif pop_size_2 == 0:
        population = draw_from_trunc_norm(18, 85, coeffs.loc['mu1', 'estimate'], coeffs.loc['sigma1', 'estimate'],
                                          pop_size_1)
    else:
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


def calculate_in_care_cd4n(pop, coeffs, knots, year):
    """ Calculate time varying cd4n as a linear function of age_cat, cd4_cat, time_from_h1yy and their cross terms """

    # Calculate spline variables
    time_from_h1yy = year - pop['h1yy'].values
    time_from_h1yy_ = (np.maximum(0, time_from_h1yy - knots['p5']) ** 2 -
                       np.maximum(0, time_from_h1yy - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    time_from_h1yy__ = (np.maximum(0, time_from_h1yy - knots['p35']) ** 2 -
                        np.maximum(0, time_from_h1yy - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    time_from_h1yy___ = (np.maximum(0, time_from_h1yy - knots['p65']) ** 2 -
                         np.maximum(0, time_from_h1yy - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])

    # Calculate time varying sqrt cd4n
    time_varying_sqrtcd4n = (coeffs.loc['intercept', 'estimate'] +
                             (coeffs.loc['agecat', 'estimate'] * pop['age_cat']) +
                             (coeffs.loc['cd4cat349', 'estimate'] * pop['cd4_cat_349']) +
                             (coeffs.loc['cd4cat499', 'estimate'] * pop['cd4_cat_499']) +
                             (coeffs.loc['cd4cat500', 'estimate'] * pop['cd4_cat_500']) +
                             (coeffs.loc['time_from_h1yy', 'estimate'] * time_from_h1yy) +
                             (coeffs.loc['_time_from_h1yy', 'estimate'] * time_from_h1yy_) +
                             (coeffs.loc['__time_from_h1yy', 'estimate'] * time_from_h1yy__) +
                             (coeffs.loc['___time_from_h1yy', 'estimate'] * time_from_h1yy___) +
                             (coeffs.loc['_timecd4cat349', 'estimate'] * time_from_h1yy_ * pop['cd4_cat_349']) +
                             (coeffs.loc['_timecd4cat499', 'estimate'] * time_from_h1yy_ * pop['cd4_cat_499']) +
                             (coeffs.loc['_timecd4cat500', 'estimate'] * time_from_h1yy_ * pop['cd4_cat_500']) +
                             (coeffs.loc['__timecd4cat349', 'estimate'] * time_from_h1yy__ * pop['cd4_cat_349']) +
                             (coeffs.loc['__timecd4cat499', 'estimate'] * time_from_h1yy__ * pop['cd4_cat_499']) +
                             (coeffs.loc['__timecd4cat500', 'estimate'] * time_from_h1yy__ * pop['cd4_cat_500']) +
                             (coeffs.loc['___timecd4cat349', 'estimate'] * time_from_h1yy___ * pop['cd4_cat_349']) +
                             (coeffs.loc['___timecd4cat499', 'estimate'] * time_from_h1yy___ * pop['cd4_cat_499']) +
                             (coeffs.loc['___timecd4cat500', 'estimate'] * time_from_h1yy___ * pop['cd4_cat_500']))

    return time_varying_sqrtcd4n


def calculate_out_care_cd4n(pop, coeffs, year):
    """ Calculate new time varying cd4 count for population out of care """
    time_out = year - pop['ltfu_year'].values
    diff = (coeffs.loc['time_out_of_naaccord', 'estimate'] * time_out) + (
                coeffs.loc['sqrtcd4n_exit', 'estimate'] * pop['sqrtcd4n_exit'].values) + coeffs.loc[
               'intercept', 'estimate']
    time_varying_sqrtcd4n = np.sqrt((pop['sqrtcd4n_exit'].values ** 2) * np.exp(diff) * 1.5)

    return time_varying_sqrtcd4n


def calculate_ltfu_prob(pop, coeffs, knots, year):
    """ Calculate the probability of loss to follow up """

    # Calculate spline variables
    age = pop['age'].values
    age_ = (np.maximum(0, age - knots['p5']) ** 2 -
            np.maximum(0, age - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    age__ = (np.maximum(0, age - knots['p35']) ** 2 -
             np.maximum(0, age - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])
    age___ = (np.maximum(0, age - knots['p65']) ** 2 -
              np.maximum(0, age - knots['p95']) ** 2) / (knots['p95'] - knots['p5'])

    # Create haart_period variable
    haart_period = (pop['h1yy'].values > 2010).astype(int)

    # Calculate log odds
    odds = (coeffs.loc['intercept', 'estimate'] +
            (coeffs.loc['age', 'estimate'] * age) +
            (coeffs.loc['_age', 'estimate'] * age_) +
            (coeffs.loc['__age', 'estimate'] * age__) +
            (coeffs.loc['___age', 'estimate'] * age___) +
            (coeffs.loc['year', 'estimate'] * year) +
            (coeffs.loc['sqrt_cd4n', 'estimate'] * pop['init_sqrtcd4n']) +
            (coeffs.loc['haart_period', 'estimate'] * haart_period))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob


def calculate_death_in_care_prob(pop, coeffs, flag, vcov, rand):
    """ Calculate the individual probability of dying in care """
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


def calculate_death_out_care_prob(pop, coeffs, year):
    """ Calculate the individual probability of dying in care """
    odds = (coeffs.loc['intercept', 'estimate'] +
            (coeffs.loc['age_cat', 'estimate'] * pop['age_cat']) +
            (coeffs.loc['tv_sqrtcd4n', 'estimate'] * pop['time_varying_sqrtcd4n']) +
            (coeffs.loc['year', 'estimate'] * year))

    # Convert to probability
    prob = np.exp(odds) / (1.0 + np.exp(odds))
    return prob


def make_pop_2009(on_art_2009, age_in_2009, h1yy_by_age_2009, cd4n_by_h1yy_2009, cd4_increase, cd4_increase_knots,
                  group_name):
    """ Create initial 2009 population. Draw ages from a mixed normal distribution truncated at 18 and 85. h1yy is
    assigned using proportions from NA-ACCORD data. Finally, sqrt cd4n is drawn from a 0-truncated normal for each
    h1yy """

    # Draw ages from the truncated mixed gaussian
    age_in_2009.loc['lambda2', 'estimate'] = 1.0 - age_in_2009.loc['lambda1', 'estimate']
    pop_size = on_art_2009[0].astype('int')
    population = sim_pop(age_in_2009, pop_size)

    # Create age categories
    population['age'] = np.floor(population['age'])
    population['age_cat'] = np.floor(population['age'] / 10)
    population.loc[population['age_cat'] > 7, 'age_cat'] = 7
    population['id'] = range(population.index.size)
    population = population.sort_values('age')
    population = population.set_index(['age_cat', 'id'])

    # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
    for age_cat, grouped in population.groupby('age_cat'):
        if h1yy_by_age_2009.index.isin([(group_name, age_cat, 2000.0)]).any():
            h1yy_data = h1yy_by_age_2009.loc[(group_name, age_cat)]
        else:  # replace missing data
            h1yy_data = h1yy_by_age_2009.loc[('msm_white_male', age_cat)]
        population.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data.index.values, size=grouped.shape[0],
                                                           p=h1yy_data.pct.values)

    # Pull cd4 count coefficients
    mean_intercept = cd4n_by_h1yy_2009['meanint']
    mean_slope = cd4n_by_h1yy_2009['meanslp']
    std_intercept = cd4n_by_h1yy_2009['stdint']
    std_slope = cd4n_by_h1yy_2009['stdslp']

    # Reindex for group operation
    population['h1yy'] = population['h1yy'].astype(int)
    population = population.reset_index().set_index(['h1yy', 'id']).sort_index()

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    for h1yy, group in population.groupby(level=0):
        mu = mean_intercept + (h1yy * mean_slope)
        sigma = std_intercept + (h1yy * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.inf, mu, sigma, size)
        population.loc[(h1yy,), 'init_sqrtcd4n'] = sqrt_cd4n

    population = population.reset_index().set_index('id').sort_index()

    # Toss out age_cat < 2
    population.loc[population['age_cat'] < 2, 'age_cat'] = 2

    # Calculate time varying cd4 count
    population = set_cd4_cat(population)
    population['h1yy_orig'] = population['h1yy']
    population['init_sqrtcd4n_orig'] = population['init_sqrtcd4n']
    population['time_varying_sqrtcd4n'] = calculate_in_care_cd4n(population.copy(), cd4_increase, cd4_increase_knots,
                                                                 2009)

    # Add final columns used for calculations and output
    population['n_lost'] = 0
    population['years_out'] = 0
    population['year_died'] = np.nan
    population['sqrtcd4n_exit'] = 0
    population['ltfu_year'] = 0
    population['intercept'] = 1.0
    population['year'] = 2009

    # Set status to 1 = 'in_care'
    population['status'] = IN_CARE

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population


def simulate_new_dx(new_dx, dx_interval):
    """ Draw number of new diagnoses from a uniform distribution between upper and lower bounds. Calculate number of
    new art initiators by assuming 75% link in the first year, then another 10% over the next three years. Assume 75%
    of these initiate art """

    # Draw new dx from a uniform distribution between upper and lower for 2016-2030 
    dx_interval = dx_interval[dx_interval.index > 2015].copy()
    dx_interval['rand'] = np.random.uniform(size=len(dx_interval.index))
    dx_interval['n_dx'] = dx_interval['lower'] + (dx_interval['upper'] - dx_interval['lower']) * dx_interval['rand']
    new_dx = pd.concat([new_dx, dx_interval.filter(items=['n_dx'])])
    new_dx = np.floor(new_dx)

    # We assume 75% link to care in the first year and a further 10% link in the next 3 years with equal probability
    new_dx['lag_step'] = new_dx['n_dx'] * 0.1 / 3.0
    new_dx['year0'] = new_dx['n_dx'] * 0.75
    new_dx['year1'] = new_dx['lag_step'].shift(1, fill_value=0)
    new_dx['year2'] = new_dx['lag_step'].shift(2, fill_value=0)
    new_dx['year3'] = new_dx['lag_step'].shift(3, fill_value=0)
    new_dx['total_linked'] = new_dx['year0'] + new_dx['year1'] + new_dx['year2'] + new_dx['year3']

    new_dx['n_art_init'] = (new_dx['total_linked'] * 0.75).astype(int)

    return new_dx.filter(items=['n_art_init'])


def make_new_population(art_init_sim, age_by_h1yy, cd4n_by_h1yy, pop_size_2009, group_name, replication):
    """ Draw ages for new art initiators """

    # Replace negative values with 0
    age_by_h1yy[age_by_h1yy < 0] = 0

    # Split into before and after 2018
    sim_coeff = age_by_h1yy.loc[age_by_h1yy.index.get_level_values('h1yy') >= 2018].copy()
    observed_coeff = age_by_h1yy.loc[age_by_h1yy.index.get_level_values('h1yy') < 2018].copy().rename(
        columns={'pred': 'estimate'})
    observed_coeff = pd.pivot_table(observed_coeff.reset_index(), values='estimate', index='h1yy',
                                    columns='param').rename_axis(None, axis=1)

    # Pull coefficients in 2018
    sim_coeff['pred18'] = np.nan
    for name, group in sim_coeff.groupby('param'):
        sim_coeff.loc[(name,), 'pred18'] = sim_coeff.loc[(name, 2018), 'pred']

        # Draw uniformly between predicted coeffs and coeff in 2018
    sim_coeff['rand'] = np.random.rand(len(sim_coeff.index))
    sim_coeff['min'] = np.minimum(sim_coeff['pred'], sim_coeff['pred18'])
    sim_coeff['max'] = np.maximum(sim_coeff['pred'], sim_coeff['pred18'])
    sim_coeff['estimate'] = sim_coeff['min'] + sim_coeff['rand'] * (sim_coeff['max'] - sim_coeff['min'])

    # Reorganize table and glue them back together
    sim_coeff = pd.pivot_table(sim_coeff.reset_index(), values='estimate', index='h1yy', columns='param').rename_axis(
        None, axis=1)
    sim_coeff = pd.concat([observed_coeff, sim_coeff])

    # Lambdas should add to 1.0
    sim_coeff.loc[sim_coeff['lambda2'] < 0, 'lambda2'] = 0
    sim_coeff.loc[sim_coeff['lambda2'] > 1, 'lambda2'] = 1
    sim_coeff['lambda1'] = 1.0 - sim_coeff['lambda2']

    # Don't simulate new dxs in 2009
    sim_coeff = sim_coeff.drop(2009)

    # Convert to long data set
    sim_coeff = gather(sim_coeff.reset_index(), key='term', value='estimate',
                       cols=['mu1', 'mu2', 'lambda1', 'lambda2', 'sigma1', 'sigma2']).set_index(['h1yy', 'term'])

    # Write out data for plots
    #feather.write_dataframe(sim_coeff.reset_index(), f'{os.getcwd()}/../../out/age_by_h1yy/{group_name}_{replication}.feather')
    #feather.write_dataframe(art_init_sim.reset_index(), f'{os.getcwd()}/../../out/art_init_sim/{group_name}_{replication}.feather')

    population = pd.DataFrame()
    for h1yy, coeffs in sim_coeff.groupby('h1yy'):
        grouped_pop = sim_pop(sim_coeff.loc[h1yy], art_init_sim.loc[h1yy, 'n_art_init'])
        grouped_pop['h1yy'] = h1yy
        population = pd.concat([population, grouped_pop])

    population['age'] = np.floor(population['age'])
    population['age_cat'] = np.floor(population['age'] / 10)
    population.loc[population['age_cat'] < 2, 'age_cat'] = 2
    population.loc[population['age_cat'] > 7, 'age_cat'] = 7

    # Add id number
    population['id'] = np.arange(pop_size_2009, (pop_size_2009 + population.index.size))

    # Pull cd4 count coefficients
    mean_intercept = cd4n_by_h1yy['meanint']
    mean_slope = cd4n_by_h1yy['meanslp']
    std_intercept = cd4n_by_h1yy['stdint']
    std_slope = cd4n_by_h1yy['stdslp']

    # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 using 
    population = population.set_index('h1yy')
    for h1yy, group in population.groupby(level=0):
        h1yy_mod = np.where(h1yy >= 2020, 2020, h1yy)  # Pin coefficients at 2020
        mu = mean_intercept + (h1yy_mod * mean_slope)
        sigma = std_intercept + (h1yy_mod * std_slope)
        size = group.shape[0]
        sqrt_cd4n = draw_from_trunc_norm(0, np.inf, mu, sigma, size)
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

    # Set status to UNINITIATED
    population['status'] = UNINITIATED

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population


###############################################################################
# Parameter and Statistics Classes                                            #
###############################################################################

class Parameters:
    def __init__(self, path, group_name, age_in_2009_flag, mortality_in_care_flag):
        with pd.HDFStore(path) as store:
            self.new_dx = store['new_dx'].loc[group_name]
            self.new_dx_interval = store['new_dx_interval'].loc[group_name]
            self.on_art_2009 = store['on_art_2009'].loc[group_name]
            self.age_in_2009 = store['age_in_2009'].loc[group_name]
            self.new_age_in_2009 = store['new_age_in_2009'].loc[group_name]
            self.h1yy_by_age_2009 = store['h1yy_by_age_2009']
            self.cd4n_by_h1yy_2009 = store['cd4n_by_h1yy_2009'].loc[group_name]
            self.age_by_h1yy = store['age_by_h1yy'].loc[group_name]
            self.cd4n_by_h1yy = store['cd4n_by_h1yy'].loc[group_name]
            self.mortality_in_care = store['mortality_in_care'].loc[group_name]
            self.mortality_in_care_vcov = store['mortality_in_care_vcov'].loc[group_name]
            self.mortality_in_care_flag = mortality_in_care_flag
            self.mortality_in_care_rand = np.random.rand()
            #print(self.mortality_in_care)
            #print(self.mortality_in_care_vcov)
            self.mortality_out_care = store['mortality_out_care'].loc[group_name]
            self.loss_to_follow_up = store['loss_to_follow_up'].loc[group_name]
            self.ltfu_knots = store['ltfu_knots'].loc[group_name]
            self.cd4_decrease = store['cd4_decrease'].loc[group_name]
            self.cd4_increase = store['cd4_increase'].loc[group_name]
            self.cd4_increase_knots = store['cd4_increase_knots'].loc[group_name]
            self.prob_reengage = store['prob_reengage'].loc[group_name]

            if age_in_2009_flag:
                rand = np.random.rand(len(self.age_in_2009.index))
                self.age_in_2009['estimate'] = (rand * (self.age_in_2009['conf_high'] - self.age_in_2009['conf_low'])) + self.age_in_2009['conf_low']


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


def output_reindex(df):
    """ Helper function for reindexing output tables """
    return df.reindex(pd.MultiIndex.from_product([df.index.levels[0], np.arange(2.0, 8.0)], names=['year', 'age_cat']),
                      fill_value=0)


###############################################################################
# Pearl Class                                                                 #
###############################################################################

class Pearl:
    def __init__(self, parameters, group_name, replication, verbose=False, cd4_reset=True):
        self.out_dir = os.path.realpath(f'{os.getcwd()}/../../out/raw')
        self.group_name = group_name
        self.replication = replication
        self.cd4_reset = cd4_reset
        self.verbose = verbose
        self.year = 2009
        self.parameters = parameters

        # Simulate number of new art initiators
        art_init_sim = simulate_new_dx(parameters.new_dx, parameters.new_dx_interval)

        # Create 2009 population
        self.population = make_pop_2009(parameters.on_art_2009, parameters.age_in_2009.copy(),
                                        parameters.h1yy_by_age_2009, parameters.cd4n_by_h1yy_2009,
                                        parameters.cd4_increase, parameters.cd4_increase_knots, group_name)

        # Create population of new art initiators
        self.population = self.population.append(
            make_new_population(art_init_sim, parameters.age_by_h1yy, parameters.cd4n_by_h1yy,
                                len(self.population.index), self.group_name, self.replication))

        # Allow loss to follow up to occur in initial year
        self.lose_to_follow_up()

        # Initiate output class
        self.stats = Statistics()

        # First recording of stats
        self.record_stats()

        # Move populations
        self.append_new()

        if self.verbose:
            print(self)

        # Move to 2010
        self.year += 1

        # Run
        self.run()

    def __str__(self):
        total = len(self.population.index)
        in_care = len(self.population.loc[self.population['status'] == IN_CARE])
        out_care = len(self.population.loc[self.population['status'] == OUT_CARE])
        dead_in_care = len(self.population.loc[self.population['status'] == DEAD_IN_CARE])
        dead_out_care = len(self.population.loc[self.population['status'] == DEAD_OUT_CARE])
        uninitiated = len(self.population.loc[self.population['status'] == UNINITIATED])

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
        alive_and_initiated = self.population['status'].isin([IN_CARE, OUT_CARE])
        self.population.loc[alive_and_initiated, 'age'] += 1
        self.population['age_cat'] = np.floor(self.population['age'] / 10)
        self.population.loc[self.population['age_cat'] < 2, 'age_cat'] = 2
        self.population.loc[self.population['age_cat'] > 7, 'age_cat'] = 7

        # Increment number of years out
        out_care = self.population['status'] == OUT_CARE
        self.population.loc[out_care, 'years_out'] += 1

    def increase_cd4_count(self):
        in_care = self.population['status'] == IN_CARE
        self.population.loc[in_care, 'time_varying_sqrtcd4n'] = calculate_in_care_cd4n(
            self.population.loc[in_care].copy(),
            self.parameters.cd4_increase,
            self.parameters.cd4_increase_knots, self.year)

    def decrease_cd4_count(self):
        out_care = self.population['status'] == OUT_CARE
        self.population.loc[out_care, 'time_varying_sqrtcd4n'] = calculate_out_care_cd4n(
            self.population.loc[out_care].copy(),
            self.parameters.cd4_decrease, self.year)

    def add_new_dx(self):
        new_dx = (self.population['status'] == UNINITIATED) & (self.population['h1yy_orig'] == self.year)
        self.population.loc[new_dx, 'status'] = IN_CARE

    def kill_in_care(self):
        in_care = self.population['status'] == IN_CARE
        coeff_matrix = self.parameters.mortality_in_care.to_numpy()
        pop_matrix = self.population[['intercept', 'year', 'age_cat', 'init_sqrtcd4n', 'h1yy']].to_numpy()
        death_prob = calculate_death_in_care_prob(pop_matrix, coeff_matrix, self.parameters.mortality_in_care_flag, self.parameters.mortality_in_care_vcov.to_numpy(), self.parameters.mortality_in_care_rand)
        print(death_prob)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & in_care
        self.population.loc[died, 'status'] = DEAD_IN_CARE
        self.population.loc[died, 'year_died'] = self.year

    def kill_out_care(self):
        out_care = self.population['status'] == OUT_CARE
        death_prob = calculate_death_out_care_prob(self.population.copy(), self.parameters.mortality_out_care,
                                                   self.year)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & out_care
        self.population.loc[died, 'status'] = DEAD_OUT_CARE
        self.population.loc[died, 'year_died'] = self.year

    def lose_to_follow_up(self):
        in_care = self.population['status'] == IN_CARE
        ltfu_prob = calculate_ltfu_prob(self.population.copy(), self.parameters.loss_to_follow_up,
                                        self.parameters.ltfu_knots, self.year)
        lost = (ltfu_prob > np.random.rand(len(self.population.index))) & in_care
        self.population.loc[lost, 'status'] = LTFU
        self.population.loc[lost, 'sqrtcd4n_exit'] = self.population.loc[lost, 'time_varying_sqrtcd4n']
        self.population.loc[lost, 'ltfu_year'] = self.year
        self.population.loc[lost, 'n_lost'] += 1

    def reengage(self):
        out_care = self.population['status'] == OUT_CARE
        reengaged = (np.random.rand(len(self.population.index)) < np.full(len(self.population.index),
                                                                          self.parameters.prob_reengage)) & out_care
        self.population.loc[reengaged, 'status'] = REENGAGED

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        if self.cd4_reset:
            self.population = set_cd4_cat(self.population)
            self.population.loc[reengaged, 'init_sqrtcd4n'] = self.population.loc[reengaged, 'time_varying_sqrtcd4n']
            self.population.loc[reengaged, 'h1yy'] = self.year

    def append_new(self):
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU

        self.population.loc[reengaged, 'status'] = IN_CARE
        self.population.loc[ltfu, 'status'] = OUT_CARE

    def record_stats(self):
        uninitiated = self.population['status'] == UNINITIATED
        in_care = self.population['status'] == IN_CARE
        out_care = self.population['status'] == OUT_CARE
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU

        # Count of new initiators by year
        if self.year == 2009:
            self.stats.new_init_count = (
                self.population.loc[uninitiated].groupby(['h1yy_orig']).size().reset_index(name='n').
                assign(replication=self.replication, group=self.group_name))

            self.stats.new_init_age = (
                self.population.loc[uninitiated].groupby(['h1yy_orig', 'age']).size().reset_index(name='n').
                assign(replication=self.replication, group=self.group_name))

            # Count of those in care by age_cat and year
            in_care_count = (self.population.loc[in_care | ltfu].groupby(['age_cat']).size()
                             .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                             .assign(year=self.year, replication=self.replication, group=self.group_name))
            self.stats.in_care_count = self.stats.in_care_count.append(in_care_count)

            # Count of those in care by age and year
            in_care_age = (self.population.loc[in_care | ltfu].groupby(['age']).size().reset_index(name='n')
                           .assign(year=self.year, replication=self.replication, group=self.group_name))
            self.stats.in_care_age = self.stats.in_care_age.append(in_care_age)

            # Count of those lost to care by age_cat and year
            ltfu_count = (self.population.loc[ltfu].groupby(['age_cat']).size()
                          .reindex(index=np.arange(2.0, 8.0), fill_value=0).reset_index(name='n')
                          .assign(year=(self.year + 1), replication=self.replication, group=self.group_name))
            self.stats.ltfu_count = self.stats.ltfu_count.append(ltfu_count)

            # Count of those lost to care by age and year
            ltfu_age = (self.population.loc[ltfu].groupby(['age']).size().reset_index(name='n')
                        .assign(year=(self.year + 1), replication=self.replication, group=self.group_name))
            self.stats.ltfu_age = self.stats.ltfu_age.append(ltfu_age)

        else:
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

    def record_final_stats(self):
        dead_in_care = self.population['status'] == DEAD_IN_CARE
        dead_out_care = self.population['status'] == DEAD_OUT_CARE

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

    def run(self):
        """ Simulate from 2010 to 2030 """
        while self.year <= 2030:

            # Everybody ages
            self.increment_age()

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


###############################################################################
# Main Function                                                               #
###############################################################################

#if __name__ == '__main__':
#    # Add argument parsing
#    parser = argparse.ArgumentParser(description='Run the PEARL model for a given group and replication')
#    parser.add_argument('param_file', help='Relative path to parameter file')
#    parser.add_argument('group', help='Risk group, e.g. msm_white_male')
#    parser.add_argument('replication', help='Replication number')
#    args = parser.parse_args()
#
#    param_file_path = os.path.realpath(f'{os.getcwd()}/{args.param_file}')
#
#    parameters = Parameters(param_file_path, args.group)
#    pearl = Pearl(parameters, args.group, args.replication, verbose=False, cd4_reset=True)
