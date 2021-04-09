# Imports
import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
pd.set_option('display.max_rows', None)

# Status Constants
ART_NAIVE = 0
DELAYED = 1
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

# Comorbidity stages
STAGE0 = ['hcv', 'smoking']
STAGE1 = ['anx', 'dpr']
STAGE2 = ['ckd', 'lipid', 'dm', 'ht']
STAGE3 = ['malig', 'esld', 'mi']

AGES = np.arange(18, 87)
AGE_CATS = np.arange(2, 8)
SIMULATION_YEARS = np.arange(2010, 2031)
ALL_YEARS = np.arange(2000, 2031)


###############################################################################
# Functions                                                                   #
###############################################################################

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
                              pop['last_init_sqrtcd4n'].ge(np.sqrt(200.0)) &
                              pop['last_init_sqrtcd4n'].lt(np.sqrt(350.0)),
                              pop['last_init_sqrtcd4n'].ge(np.sqrt(350.0)) &
                              pop['last_init_sqrtcd4n'].lt(np.sqrt(500.0)),
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
    pop_matrix = pop[['intercept', 'time_from_h1yy', 'time_from_h1yy_', 'time_from_h1yy__', 'time_from_h1yy___', 'cd4_cat_349', 'cd4_cat_499', 'cd4_cat_500', 'age_cat', 'timecd4cat349_',
                      'timecd4cat499_', 'timecd4cat500_', 'timecd4cat349__', 'timecd4cat499__', 'timecd4cat500__', 'timecd4cat349___', 'timecd4cat499___', 'timecd4cat500___']].to_numpy(dtype=float)

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
    pop_matrix = pop[['intercept', 'time_out', 'sqrtcd4n_exit']].to_numpy(dtype=float)
    diff = np.matmul(pop_matrix, coeffs)

    if sa is not None:
        se = np.sqrt(np.sum(np.matmul(pop_matrix, vcov) * pop_matrix, axis=1))
        low = diff - 1.96 * se
        high = diff + 1.96 * se
        diff = (sa * (high - low)) + low

    new_cd4 = np.sqrt((pop['sqrtcd4n_exit'].to_numpy(dtype=float) ** 2) * np.exp(diff) * SMEARING)
    return new_cd4


def create_mortality_pop_matrix(pop, comorbidity_flag, in_care_flag, parameters):

    if in_care_flag:
        if comorbidity_flag:
            pop['delta_bmi_'] = restricted_cubic_spline_var(pop['delta_bmi'], parameters.mortality_in_care_delta_bmi, 1)
            pop['delta_bmi__'] = restricted_cubic_spline_var(pop['delta_bmi'], parameters.mortality_in_care_delta_bmi, 2)
            pop['post_art_bmi_'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_in_care_post_art_bmi, 1)
            pop['post_art_bmi__'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_in_care_post_art_bmi, 2)
            return pop[['age_cat', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'ckd',
                        'dm', 'dpr', 'esld', 'h1yy', 'hcv', 'ht', 'intercept', 'lipid', 'malig', 'mi', 'smoking', 'init_sqrtcd4n', 'year']].to_numpy(dtype=float)
        else:
            pop['years_art'] = pop['year'] - pop['h1yy']
            pop['years_art_'] = restricted_cubic_spline_var(pop['years_art'], parameters.mortality_in_care_years_art, 1)
            pop['years_art__'] = restricted_cubic_spline_var(pop['years_art'], parameters.mortality_in_care_years_art, 2)
            pop['age_'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_in_care_age, 1)
            pop['age__'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_in_care_age, 2)
            pop['init_sqrtcd4n_'] = restricted_cubic_spline_var(pop['init_sqrtcd4n'], parameters.mortality_in_care_sqrtcd4, 1)
            pop['init_sqrtcd4n__'] = restricted_cubic_spline_var(pop['init_sqrtcd4n'], parameters.mortality_in_care_sqrtcd4, 2)
            pop['year_'] = pop['year'] if pop['year'].iloc[0] <= 2018 else 2018
            return pop[['age', 'age_', 'age__', 'intercept', 'init_sqrtcd4n_', 'init_sqrtcd4n__', 'init_sqrtcd4n', 'years_art', 'years_art_', 'years_art__', 'year_']].to_numpy(dtype=float)
    else:
        if comorbidity_flag:
            pop['delta_bmi_'] = restricted_cubic_spline_var(pop['delta_bmi'], parameters.mortality_out_care_delta_bmi, 1)
            pop['delta_bmi__'] = restricted_cubic_spline_var(pop['delta_bmi'], parameters.mortality_out_care_delta_bmi, 2)
            pop['post_art_bmi_'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_out_care_post_art_bmi, 1)
            pop['post_art_bmi__'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_out_care_post_art_bmi, 2)
            return pop[['age_cat', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'ckd',
                        'dm', 'dpr', 'esld', 'hcv', 'ht', 'intercept', 'lipid', 'malig', 'mi', 'smoking', 'time_varying_sqrtcd4n', 'year']].to_numpy(dtype=float)
        else:
            pop['age_'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_out_care_age, 1)
            pop['age__'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_out_care_age, 2)
            pop['time_varying_sqrtcd4n_'] = restricted_cubic_spline_var(pop['time_varying_sqrtcd4n'], parameters.mortality_out_care_tv_sqrtcd4, 1)
            pop['time_varying_sqrtcd4n__'] = restricted_cubic_spline_var(pop['time_varying_sqrtcd4n'], parameters.mortality_out_care_tv_sqrtcd4, 2)
            pop['year_'] = pop['year'] if pop['year'].iloc[0] <= 2018 else 2018
            return pop[['age', 'age_', 'age__', 'intercept', 'time_varying_sqrtcd4n', 'time_varying_sqrtcd4n_', 'time_varying_sqrtcd4n__', 'year_']].to_numpy(dtype=float)


def create_comorbidity_pop_matrix(pop, condition, delta_bmi_knots=pd.DataFrame(), post_art_bmi_knots=pd.DataFrame()):
    pop['time_since_art'] = pop['year'] - pop['h1yy']
    pop['out_care'] = (pop['status'] == ART_NONUSER).astype(int)
    if not delta_bmi_knots.empty:
        pop['delta_bmi_'] = restricted_cubic_spline_var(pop['delta_bmi'], delta_bmi_knots, 1)
        pop['delta_bmi__'] = restricted_cubic_spline_var(pop['delta_bmi'], delta_bmi_knots, 2)
        pop['post_art_bmi_'] = restricted_cubic_spline_var(pop['post_art_bmi'], post_art_bmi_knots, 1)
        pop['post_art_bmi__'] = restricted_cubic_spline_var(pop['post_art_bmi'], post_art_bmi_knots, 2)

    if condition == 'anx':
        return pop[['age', 'init_sqrtcd4n', 'dpr', 'time_since_art', 'hcv', 'intercept', 'out_care', 'smoking', 'year']].to_numpy(dtype=float)

    elif condition == 'dpr':
        return pop[['age', 'anx', 'init_sqrtcd4n', 'time_since_art', 'hcv', 'intercept', 'out_care', 'smoking', 'year']].to_numpy(dtype=float)

    elif condition == 'ckd':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'dm', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy(dtype=float)

    elif condition == 'lipid':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dm', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'out_care', 'smoking', 'year']].to_numpy(dtype=float)

    elif condition == 'dm':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy(dtype=float)

    elif condition == 'ht':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dm', 'dpr', 'time_since_art', 'hcv', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy(dtype=float)

    elif condition in ['malig', 'esld', 'mi']:
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dm', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'lipid', 'out_care', 'smoking', 'year']].to_numpy(dtype=float)


def restricted_cubic_spline_var(x, t, i):
    # https:github.com/harrelfe/Hmisc/blob/master/R/rcspline.eval.s
    kd = (t[3] - t[0]) ** (2 / 3)
    y = (np.maximum(0, (x - t[i - 1]) / kd) ** 3
         - (np.maximum(0, (x - t[2]) / kd) ** 3) * (t[3] - t[i - 1]) / (t[3] - t[2])
         + (np.maximum(0, (x - t[3]) / kd) ** 3) * (t[2] - t[i - 1]) / (t[3] - t[2]))
    return y


def calculate_pre_art_bmi(pop, model, coeffs, t_age, t_h1yy, initial_pop=False):
    if initial_pop:
        pop['age_init'] = (pop['age'] - pop['year'] + pop['h1yy'])
    else:
        pop['age_init'] = pop['age']

    pre_art_bmi = np.nan
    if model == 6:
        pop['age_'] = restricted_cubic_spline_var(pop['age_init'], t_age, 1)
        pop['age__'] = restricted_cubic_spline_var(pop['age_init'], t_age, 2)
        h1yy = pop['h1yy'].values
        pop['h1yy_'] = restricted_cubic_spline_var(h1yy, t_h1yy, 1)
        pop['h1yy__'] = restricted_cubic_spline_var(h1yy, t_h1yy, 2)
        pop_matrix = pop[['age_init', 'age_', 'age__', 'h1yy', 'h1yy_', 'h1yy__', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
        pre_art_bmi = 10.0 ** log_pre_art_bmi

    elif model == 5:
        pop['age_'] = restricted_cubic_spline_var(pop['age_init'], t_age, 1)
        pop['age__'] = restricted_cubic_spline_var(pop['age_init'], t_age, 2)
        pop_matrix = pop[['age_init', 'age_', 'age__', 'h1yy', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
        pre_art_bmi = 10.0 ** log_pre_art_bmi

    elif model == 3:
        pop_matrix = pop[['age_init', 'h1yy', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
        pre_art_bmi = 10.0 ** log_pre_art_bmi

    elif model == 2:
        pop['age_'] = (pop['age_init'] >= 30) & (pop['age_init'] < 40)
        pop['age__'] = (pop['age_init'] >= 40) & (pop['age_init'] < 50)
        pop['age___'] = (pop['age_init'] >= 50) & (pop['age_init'] < 60)
        pop['age____'] = pop['age_init'] >= 60
        h1yy = pop['h1yy'].values
        pop['h1yy_'] = restricted_cubic_spline_var(h1yy, t_h1yy, 1)
        pop['h1yy__'] = restricted_cubic_spline_var(h1yy, t_h1yy, 2)
        pop_matrix = pop[['age_', 'age__', 'age___', 'age____',
                          'h1yy', 'h1yy_', 'h1yy__', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
        pre_art_bmi = 10.0 ** log_pre_art_bmi

    elif model == 1:
        pop['age_'] = (pop['age_init'] >= 30) & (pop['age_init'] < 40)
        pop['age__'] = (pop['age_init'] >= 40) & (pop['age_init'] < 50)
        pop['age___'] = (pop['age_init'] >= 50) & (pop['age_init'] < 60)
        pop['age____'] = pop['age_init'] >= 60
        pop_matrix = pop[['age_', 'age__', 'age___', 'age____', 'h1yy', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
        pre_art_bmi = 10.0 ** log_pre_art_bmi

    return pre_art_bmi


def calculate_post_art_bmi(pop, parameters, initial_pop=False):
    coeffs = parameters.post_art_bmi
    t_age = parameters.post_art_bmi_age_knots
    t_pre_sqrt = parameters.post_art_bmi_pre_art_bmi_knots
    t_sqrtcd4 = parameters.post_art_bmi_cd4_knots
    t_sqrtcd4_post = parameters.post_art_bmi_cd4_post_knots

    if initial_pop:
        pop['age_init'] = (pop['age'] - pop['year'] + pop['h1yy'])
    else:
        pop['age_init'] = pop['age']

    pop['age_'] = restricted_cubic_spline_var(pop['age_init'], t_age, 1)
    pop['age__'] = restricted_cubic_spline_var(pop['age_init'], t_age, 2)
    pop['pre_sqrt'] = pop['pre_art_bmi'] ** 0.5
    pop['pre_sqrt_'] = restricted_cubic_spline_var(pop['pre_sqrt'], t_pre_sqrt, 1)
    pop['pre_sqrt__'] = restricted_cubic_spline_var(pop['pre_sqrt'], t_pre_sqrt, 2)
    pop['sqrtcd4'] = pop['init_sqrtcd4n']
    pop['sqrtcd4_'] = restricted_cubic_spline_var(pop['sqrtcd4'], t_sqrtcd4, 1)
    pop['sqrtcd4__'] = restricted_cubic_spline_var(pop['sqrtcd4'], t_sqrtcd4, 2)

    pop_future = pop.copy().assign(age=pop['age_init'] + 2)
    post_art_year = pop['h1yy'] + 2
    pop_future['age_cat'] = np.floor(pop_future['age'] / 10)
    pop_future.loc[pop_future['age_cat'] < 2, 'age_cat'] = 2
    pop_future.loc[pop_future['age_cat'] > 7, 'age_cat'] = 7
    pop['sqrtcd4_post'] = calculate_cd4_increase(pop_future, parameters.cd4_increase_knots, post_art_year, parameters.cd4_increase.to_numpy(dtype=float),
                                                 parameters.cd4_increase_vcov.to_numpy(dtype=float), parameters.cd4_increase_sa)
    pop['sqrtcd4_post_'] = restricted_cubic_spline_var(pop['sqrtcd4_post'], t_sqrtcd4_post, 1)
    pop['sqrtcd4_post__'] = restricted_cubic_spline_var(pop['sqrtcd4_post'], t_sqrtcd4_post, 2)
    pop_matrix = pop[['age_init', 'age_', 'age__', 'h1yy', 'intercept', 'pre_sqrt', 'pre_sqrt_', 'pre_sqrt_', 'sqrtcd4',
                      'sqrtcd4_', 'sqrtcd4__', 'sqrtcd4_post', 'sqrtcd4_post_', 'sqrtcd4_post__']].to_numpy(dtype=float)
    sqrt_post_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
    post_art_bmi = sqrt_post_art_bmi ** 2.0

    return post_art_bmi


def create_ltfu_pop_matrix(pop, knots):
    """ Create the population matrix for use in calculating probability of loss to follow up"""

    age = pop['age'].values
    pop['age_'] = (np.maximum(0, age - knots['p5']) ** 2
                   - (np.maximum(0, age - knots['p95']) ** 2)) / (knots['p95'] - knots['p5'])
    pop['age__'] = (np.maximum(0, age - knots['p35']) ** 2
                    - (np.maximum(0, age - knots['p95']) ** 2)) / (knots['p95'] - knots['p5'])
    pop['age___'] = (np.maximum(0, age - knots['p65']) ** 2
                     - (np.maximum(0, age - knots['p95']) ** 2)) / (knots['p95'] - knots['p5'])

    pop['haart_period'] = (pop['h1yy'].values > 2010).astype(int)
    return pop[['intercept', 'age', 'age_', 'age__', 'age___', 'year', 'init_sqrtcd4n', 'haart_period']].to_numpy(dtype=float)


def calculate_prob(pop, coeffs, sa, vcov):
    """ Calculate the individual probability from logistic regression """

    log_odds = np.matmul(pop, coeffs)

    if sa is not None:
        # Calculate variance of prediction using matrix multiplication
        a = np.matmul(pop, vcov)
        b = a * pop
        c = np.sum(b, axis=1)

        se = np.sqrt(c)
        low = log_odds - (1.96 * se)
        high = log_odds + (1.96 * se)
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
    new_agents = new_dx.loc[SIMULATION_YEARS, ['art_users', 'art_nonusers']]
    new_agents['total'] = new_agents['art_users'] + new_agents['art_nonusers']

    return n_initial_nonusers, new_agents


def make_pop_2009(parameters, n_initial_nonusers, group_name, replication):
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
        population.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data.index.values, size=grouped.shape[0], p=h1yy_data.pct.values)

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

    population['time_varying_sqrtcd4n'] = calculate_cd4_increase(population.copy(), parameters.cd4_increase_knots, 2009, parameters.cd4_increase.to_numpy(dtype=float),
                                                                 parameters.cd4_increase_vcov.to_numpy(dtype=float), parameters.cd4_increase_sa)

    population['post_art_sqrtcd4n'] = calculate_cd4_increase(population.copy(), parameters.cd4_increase_knots, population['h1yy'] + 1, parameters.cd4_increase.to_numpy(dtype=float),
                                                             parameters.cd4_increase_vcov.to_numpy(dtype=float), parameters.cd4_increase_sa)

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
        # Bmi
        population['pre_art_bmi'] = calculate_pre_art_bmi(population, parameters.pre_art_bmi_model, parameters.pre_art_bmi,
                                                          parameters.pre_art_bmi_age_knots, parameters.pre_art_bmi_h1yy_knots, True)
        population['post_art_bmi'] = calculate_post_art_bmi(population, parameters, True)
        population['delta_bmi'] = population['post_art_bmi'] - population['pre_art_bmi']

        # Apply comorbidities
        for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
            population[condition] = (np.random.rand(len(population.index)) < parameters.prev_users_dict[condition].values).astype(int)

        population['mm'] = population[STAGE2 + STAGE3].sum(axis=1)

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population


def make_new_population(parameters, n_new_agents, pop_size_2009, group_name, replication, stats_new_pop):
    """ Draw ages for new art initiators """

    # Draw a random value between predicted and 2018 predicted value for years greater than 2018
    rand = np.random.rand(len(parameters.age_by_h1yy.index))
    parameters.age_by_h1yy['estimate'] = (rand * (parameters.age_by_h1yy['high_value'] - parameters.age_by_h1yy['low_value'])) + parameters.age_by_h1yy['low_value']
    stats_new_pop.art_coeffs = parameters.age_by_h1yy[['estimate']].assign(group=group_name, replication=replication, variable='age').reset_index()

    rand = np.random.rand(len(parameters.cd4n_by_h1yy.index))
    parameters.cd4n_by_h1yy['estimate'] = (rand * (parameters.cd4n_by_h1yy['high_value'] - parameters.cd4n_by_h1yy['low_value'])) + parameters.cd4n_by_h1yy['low_value']
    art_coeffs_cd4 = parameters.cd4n_by_h1yy[['estimate']].assign(group=group_name, replication=replication, variable='cd4').reset_index()
    stats_new_pop.art_coeffs = pd.concat([stats_new_pop.art_coeffs, art_coeffs_cd4])

    # Create population
    population = pd.DataFrame()
    for h1yy in parameters.age_by_h1yy.index.levels[0]:
        n_users = n_new_agents.loc[h1yy, 'art_users']
        n_nonusers = n_new_agents.loc[h1yy, 'art_nonusers']
        grouped_pop = simulate_ages(parameters.age_by_h1yy.loc[h1yy], n_users + n_nonusers)
        grouped_pop['h1yy'] = h1yy
        grouped_pop['status'] = ART_NAIVE
        non_users = np.random.choice(a=len(grouped_pop.index), size=n_nonusers, replace=False)
        grouped_pop.loc[non_users, 'status'] = DELAYED

        population = pd.concat([population, grouped_pop])

    delayed = population['status'] == DELAYED
    years_out_of_care =  np.random.choice(a=parameters.years_out_of_care['years'], size=len(population.loc[delayed]), p=parameters.years_out_of_care['probability'])

    population.loc[delayed, 'h1yy'] = population.loc[delayed, 'h1yy'] + years_out_of_care
    population.loc[delayed, 'status'] = ART_NAIVE
    population = population[population['h1yy'] <= 2030].copy()

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
        population['pre_art_bmi'] = calculate_pre_art_bmi(population, parameters.pre_art_bmi_model, parameters.pre_art_bmi, parameters.pre_art_bmi_age_knots, parameters.pre_art_bmi_h1yy_knots)
        population['post_art_bmi'] = calculate_post_art_bmi(population, parameters)
        population['delta_bmi'] = population['post_art_bmi'] - population['pre_art_bmi']

        # Apply comorbidities
        for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
            population[condition] = (np.random.rand(len(population.index)) < parameters.prev_inits_dict[condition].values).astype(int)

        population['mm'] = population[STAGE2 + STAGE3].sum(axis=1)

    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)

    return population


def create_mm_detail_stats(pop):
    # Encode multimorbidity as 11 bit integer
    df = pop[['age_cat', 'smoking', 'hcv', 'anx', 'dpr', 'ckd', 'lipid', 'dm', 'ht', 'malig', 'esld', 'mi']].copy()
    df['multimorbidity'] = (df['smoking'].map(str) + df['hcv'].map(str) + df['anx'].map(str) + df['dpr'].map(str)
                            + df['ckd'].map(str) + df['lipid'].map(str) + df['dm'].map(str) + df['ht'].map(str)
                            + df['malig'].map(str) + df['esld'].map(str) + df['mi'].map(str)).apply(int, base=2)

    # Count how many people have each unique set of comorbidities
    df = df.groupby(['multimorbidity']).size()
    df = df.reindex(index=range(2048), fill_value=0).reset_index(name='n')
    return df


###############################################################################
# Parameter and Statistics Classes                                            #
###############################################################################

class Parameters:
    def __init__(self, path, rerun_folder, group_name, replications, comorbidity_flag, mm_detail_flag, sa_dict, new_dx='base',
                 output_folder=f'{os.getcwd()}/../../out/raw', verbose=False, smoking_intervention=False):
        self.rerun_folder = rerun_folder
        self.output_folder = output_folder
        self.comorbidity_flag = comorbidity_flag
        self.mm_detail_flag = mm_detail_flag
        self.smoking_intervention = smoking_intervention
        self.verbose = verbose

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
        if lambda1_sa == 0:
            self.age_in_2009.loc['lambda1', 'estimate'] = self.age_in_2009.loc['lambda1', 'conf_low']
        elif lambda1_sa == 1:
            self.age_in_2009.loc['lambda1', 'estimate'] = self.age_in_2009.loc['lambda1', 'conf_high']

        if mu1_sa == 0:
            self.age_in_2009.loc['mu1', 'estimate'] = self.age_in_2009.loc['mu1', 'conf_low']
        elif mu1_sa == 1:
            self.age_in_2009.loc['mu1', 'estimate'] = self.age_in_2009.loc['mu1', 'conf_high']

        if mu2_sa == 0:
            self.age_in_2009.loc['mu2', 'estimate'] = self.age_in_2009.loc['mu2', 'conf_low']
        elif mu2_sa == 1:
            self.age_in_2009.loc['mu2', 'estimate'] = self.age_in_2009.loc['mu2', 'conf_high']

        if sigma1_sa == 0:
            self.age_in_2009.loc['sigma1', 'estimate'] = self.age_in_2009.loc['sigma1', 'conf_low']
        elif sigma1_sa == 1:
            self.age_in_2009.loc['sigma1', 'estimate'] = self.age_in_2009.loc['sigma1', 'conf_high']

        if sigma2_sa == 0:
            self.age_in_2009.loc['sigma2', 'estimate'] = self.age_in_2009.loc['sigma2', 'conf_low']
        elif sigma2_sa == 1:
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
        elif sa_dict['new_pop_size'] == 1:
            self.new_dx['lower'] = self.new_dx['upper']

        self.linkage_to_care = pd.read_hdf(path, 'linkage_to_care').loc[group_name]
        self.age_by_h1yy = pd.read_hdf(path, 'age_by_h1yy').loc[group_name]
        self.cd4n_by_h1yy = pd.read_hdf(path, 'cd4n_by_h1yy').loc[group_name]

        # Mortality In Care
        self.mortality_in_care = pd.read_hdf(path, 'mortality_in_care').loc[group_name]
        self.mortality_in_care_age = pd.read_hdf(path, 'mortality_in_care_age').loc[group_name]
        self.mortality_in_care_sqrtcd4 = pd.read_hdf(path, 'mortality_in_care_sqrtcd4').loc[group_name]
        self.mortality_in_care_years_art = pd.read_hdf(path, 'mortality_in_care_years_art').loc[group_name]
        #self.mortality_in_care_vcov = pd.read_hdf(path, 'mortality_in_care_vcov').loc[group_name]
        self.mortality_in_care_vcov = pd.DataFrame()
        self.mortality_in_care_sa = sa_dict['mortality_in_care']

        # Mortality Out Of Care
        self.mortality_out_care = pd.read_hdf(path, 'mortality_out_care').loc[group_name]
        self.mortality_out_care_age = pd.read_hdf(path, 'mortality_out_care_age').loc[group_name]
        self.mortality_out_care_tv_sqrtcd4 = pd.read_hdf(path, 'mortality_out_care_tv_sqrtcd4').loc[group_name]
        #self.mortality_out_care_vcov = pd.read_hdf(path, 'mortality_out_care_vcov').loc[group_name]
        self.mortality_out_care_vcov = pd.DataFrame()
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

        # BMI
        self.pre_art_bmi = pd.read_hdf(path, 'pre_art_bmi').loc[group_name]
        self.pre_art_bmi_model = pd.read_hdf(path, 'pre_art_bmi_model').loc[group_name].values[0]
        self.pre_art_bmi_age_knots = pd.read_hdf(path, 'pre_art_bmi_age_knots').loc[group_name]
        self.pre_art_bmi_h1yy_knots = pd.read_hdf(path, 'pre_art_bmi_h1yy_knots').loc[group_name]
        self.post_art_bmi = pd.read_hdf(path, 'post_art_bmi').loc[group_name]
        self.post_art_bmi_age_knots = pd.read_hdf(path, 'post_art_bmi_age_knots').loc[group_name]
        self.post_art_bmi_pre_art_bmi_knots = pd.read_hdf(path, 'post_art_bmi_pre_art_bmi_knots').loc[group_name]
        self.post_art_bmi_cd4_knots = pd.read_hdf(path, 'post_art_bmi_cd4_knots').loc[group_name]
        self.post_art_bmi_cd4_post_knots = pd.read_hdf(path, 'post_art_bmi_cd4_post_knots').loc[group_name]

        # Comorbidities
        self.prev_users_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_prev_users').loc[group_name] for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3}
        self.prev_inits_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_prev_inits').loc[group_name] for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3}
        self.comorbidity_coeff_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_coeff').loc[group_name] for comorbidity in STAGE1 + STAGE2 + STAGE3}
        self.delta_bmi_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_delta_bmi').loc[group_name] for comorbidity in STAGE2 + STAGE3}
        self.post_art_bmi_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_post_art_bmi').loc[group_name] for comorbidity in STAGE2 + STAGE3}

        # Comortality
        self.mortality_in_care_co = pd.read_hdf(path, 'mortality_in_care_co').loc[group_name]
        self.mortality_in_care_delta_bmi = pd.read_hdf(path, 'mortality_in_care_delta_bmi').loc[group_name]
        self.mortality_in_care_post_art_bmi = pd.read_hdf(path, 'mortality_in_care_post_art_bmi').loc[group_name]
        self.mortality_out_care_co = pd.read_hdf(path, 'mortality_out_care_co').loc[group_name]
        self.mortality_out_care_delta_bmi = pd.read_hdf(path, 'mortality_out_care_delta_bmi').loc[group_name]
        self.mortality_out_care_post_art_bmi = pd.read_hdf(path, 'mortality_out_care_post_art_bmi').loc[group_name]


class Statistics:
    def __init__(self, out_list=None, comorbidity_flag=None, mm_detail_flag=None):
        self.comorbidity_flag = comorbidity_flag
        self.mm_detail_flag = mm_detail_flag
        self.in_care_age = pd.concat([out.in_care_age for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.out_care_age = pd.concat([out.out_care_age for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.reengaged_age = pd.concat([out.reengaged_age for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.ltfu_age = pd.concat([out.ltfu_age for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.dead_in_care_age = pd.concat([out.dead_in_care_age for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.dead_out_care_age = pd.concat([out.dead_out_care_age for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.new_init_age = pd.concat([out.new_init_age for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.years_out = pd.concat([out.years_out for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.cd4_inits = pd.concat([out.cd4_inits for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.cd4_in_care = pd.concat([out.cd4_in_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.cd4_out_care = pd.concat([out.cd4_out_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
        self.art_coeffs = pd.concat([out.art_coeffs for out in out_list], ignore_index=True) if out_list else pd.DataFrame()

        if self.comorbidity_flag:
            self.incidence_in_care = pd.concat([out.incidence_in_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.incidence_out_care = pd.concat([out.incidence_out_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.prevalence_in_care = pd.concat([out.prevalence_in_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.prevalence_out_care = pd.concat([out.prevalence_out_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.prevalence_inits = pd.concat([out.prevalence_inits for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.prevalence_dead = pd.concat([out.prevalence_dead for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.mm_in_care = pd.concat([out.mm_in_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.mm_out_care = pd.concat([out.mm_out_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.mm_inits = pd.concat([out.mm_inits for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            self.mm_dead = pd.concat([out.mm_dead for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
            if self.mm_detail_flag:
                self.mm_detail_in_care = pd.concat([out.mm_detail_in_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
                self.mm_detail_out_care = pd.concat([out.mm_detail_out_care for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
                self.mm_detail_inits = pd.concat([out.mm_detail_inits for out in out_list], ignore_index=True) if out_list else pd.DataFrame()
                self.mm_detail_dead = pd.concat([out.mm_detail_dead for out in out_list], ignore_index=True) if out_list else pd.DataFrame()

    def save(self, output_folder):
        for name, df in self.__dict__.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(f'{output_folder}/{name}.csv', index=False)



###############################################################################
# Pearl Class                                                                 #
###############################################################################

class Pearl:
    def __init__(self, parameters, group_name, replication):
        self.group_name = group_name
        self.replication = replication
        self.year = 2009
        self.parameters = parameters

        # If this is a rerun, reload the random state
        if self.parameters.rerun_folder is not None:
            with open(f'{self.parameters.rerun_folder}/random_states/{self.group_name}_{self.replication}.state', 'rb') as state_file_load, \
                    open(f'{self.parameters.output_folder}/random_states/{self.group_name}_{self.replication}.state', 'wb') as state_file_save:
                state = pickle.load(state_file_load)
                pickle.dump(state, state_file_save)
            np.random.set_state(state)
        else:
            state = np.random.get_state()
            with open(f'{self.parameters.output_folder}/random_states/{self.group_name}_{self.replication}.state', 'wb') as state_file:
                pickle.dump(state, state_file)

        # Initiate output class
        self.stats = Statistics(comorbidity_flag=self.parameters.comorbidity_flag, mm_detail_flag=self.parameters.mm_detail_flag)

        # Simulate number of new art initiators
        n_initial_nonusers, n_new_agents = simulate_new_dx(parameters.new_dx, parameters.linkage_to_care)

        # Create 2009 population
        self.population = make_pop_2009(parameters, n_initial_nonusers, self.group_name, self.replication)

        # Create population of new art initiators
        self.population = self.population.append(
            make_new_population(self.parameters, n_new_agents, len(self.population.index), self.group_name, self.replication, self.stats))

        # First recording of stats
        self.record_stats()

        # Print populations
        if self.parameters.verbose:
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
        uninitiated_user = len(self.population.loc[self.population['status'].isin([ART_NAIVE])])

        string = 'Year End: ' + str(self.year) + '\n'
        string += 'Total Population Size: ' + str(total) + '\n'
        string += 'In Care Size: ' + str(in_care) + '\n'
        string += 'Out Care Size: ' + str(out_care) + '\n'
        string += 'Dead In Care Size: ' + str(dead_in_care) + '\n'
        string += 'Dead Out Care Size: ' + str(dead_out_care) + '\n'
        string += 'Uninitiated User Size: ' + str(uninitiated_user) + '\n'
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
            self.parameters.cd4_increase.to_numpy(dtype=float),
            self.parameters.cd4_increase_vcov.to_numpy(dtype=float),
            self.parameters.cd4_increase_sa)

    def decrease_cd4_count(self):
        out_care = self.population['status'] == ART_NONUSER
        self.population.loc[out_care, 'time_varying_sqrtcd4n'] = calculate_cd4_decrease(
            self.population.loc[out_care].copy(),
            self.parameters.cd4_decrease.to_numpy(dtype=float),
            self.parameters.cd4_decrease_sa,
            self.parameters.cd4_decrease_vcov.to_numpy(dtype=float))

    def add_new_user(self):
        new_user = (self.population['status'] == ART_NAIVE) & (self.population['h1yy'] == self.year)
        self.population.loc[new_user, 'status'] = ART_USER

    def kill_in_care(self):
        in_care = self.population['status'] == ART_USER
        coeff_matrix = self.parameters.mortality_in_care_co.to_numpy(dtype=float) if self.parameters.comorbidity_flag else self.parameters.mortality_in_care.to_numpy(dtype=float)
        pop_matrix = create_mortality_pop_matrix(self.population.copy(), self.parameters.comorbidity_flag, True, self.parameters)
        vcov_matrix = self.parameters.mortality_in_care_vcov.to_numpy(dtype=float)
        death_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.mortality_in_care_sa, vcov_matrix)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & in_care
        self.population.loc[died, 'status'] = DYING_ART_USER
        self.population.loc[died, 'year_died'] = self.year

    def kill_out_care(self):
        out_care = self.population['status'] == ART_NONUSER
        coeff_matrix = self.parameters.mortality_out_care_co.to_numpy(dtype=float) if self.parameters.comorbidity_flag else self.parameters.mortality_out_care.to_numpy(dtype=float)
        pop_matrix = create_mortality_pop_matrix(self.population.copy(), self.parameters.comorbidity_flag, False, self.parameters)
        vcov_matrix = self.parameters.mortality_out_care_vcov.to_numpy(dtype=float)
        death_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.mortality_out_care_sa, vcov_matrix)
        died = ((death_prob > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & out_care
        self.population.loc[died, 'status'] = DYING_ART_NONUSER
        self.population.loc[died, 'year_died'] = self.year
        self.population.loc[died, 'return_year'] = 0

    def lose_to_follow_up(self):
        in_care = self.population['status'] == ART_USER
        coeff_matrix = self.parameters.loss_to_follow_up.to_numpy(dtype=float)
        vcov_matrix = self.parameters.loss_to_follow_up_vcov.to_numpy(dtype=float)
        pop_matrix = create_ltfu_pop_matrix(self.population.copy(), self.parameters.ltfu_knots)
        ltfu_prob = calculate_prob(pop_matrix, coeff_matrix, self.parameters.loss_to_follow_up_sa, vcov_matrix)
        lost = (ltfu_prob > np.random.rand(len(self.population.index))) & in_care
        n_lost = len(self.population.loc[lost])
        years_out_of_care = np.random.choice(a=self.parameters.years_out_of_care['years'], size=n_lost, p=self.parameters.years_out_of_care['probability'])

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
        years_out = (pd.DataFrame(self.population.loc[reengaged, 'years_out'].value_counts())
                     .reindex(range(1, 16), fill_value=0).reset_index()
                     .rename(columns={'index': 'years', 'years_out': 'n'})
                     .assign(group=self.group_name, replication=self.replication, year=self.year))
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

        for condition in STAGE1:
            coeff_matrix = self.parameters.comorbidity_coeff_dict[condition].to_numpy(dtype=float)
            pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition=condition)

            prob = calculate_prob(pop_matrix, coeff_matrix, None, None)
            rand = prob > np.random.rand(len(self.population.index))
            old = self.population[condition]
            new = rand & (in_care | out_care) & ~old

            incidence_in_care = (self.population.loc[new & in_care].groupby(['age_cat']).size()
                                 .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                 .assign(year=self.year, replication=self.replication,
                                         group=self.group_name, condition=condition))
            self.stats.incidence_in_care = self.stats.incidence_in_care.append(incidence_in_care)

            incidence_out_care = (self.population.loc[new & out_care].groupby(['age_cat']).size()
                                  .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                  .assign(year=self.year, replication=self.replication,
                                          group=self.group_name, condition=condition))
            self.stats.incidence_out_care = self.stats.incidence_out_care.append(incidence_out_care)

            self.population[condition] = (old | new).astype(int)

    def apply_stage_2_and_3(self):
        in_care = self.population['status'] == ART_USER
        out_care = self.population['status'] == ART_NONUSER

        for condition in STAGE2 + STAGE3:
            coeff_matrix = self.parameters.comorbidity_coeff_dict[condition].to_numpy(dtype=float)
            pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition,
                                                       self.parameters.delta_bmi_dict[condition],
                                                       self.parameters.post_art_bmi_dict[condition])
            prob = calculate_prob(pop_matrix, coeff_matrix, None, None)
            rand = prob > np.random.rand(len(self.population.index))
            old = self.population[condition]
            new = rand & (in_care | out_care) & ~old

            incidence_in_care = (self.population.loc[new & in_care].groupby(['age_cat']).size()
                                 .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                 .assign(year=self.year, replication=self.replication,
                                         group=self.group_name, condition=condition))
            self.stats.incidence_in_care = self.stats.incidence_in_care.append(incidence_in_care)

            incidence_out_care = (self.population.loc[new & out_care].groupby(['age_cat']).size()
                                  .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                  .assign(year=self.year, replication=self.replication,
                                          group=self.group_name, condition=condition))
            self.stats.incidence_out_care = self.stats.incidence_out_care.append(incidence_out_care)

            self.population[condition] = (old | new).astype(int)

    def update_mm(self):
        self.population['mm'] = self.population[STAGE2 + STAGE3].sum(axis=1)

    def record_stats(self):
        stay_in_care = self.population['status'] == ART_USER
        stay_out_care = self.population['status'] == ART_NONUSER
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU
        dying_art_user = self.population['status'] == DYING_ART_USER
        dying_art_nonuser = self.population['status'] == DYING_ART_NONUSER
        dying = dying_art_nonuser | dying_art_user
        in_care = stay_in_care | ltfu | dying_art_user
        out_care = stay_out_care | reengaged | dying_art_nonuser
        initiating = self.population['h1yy'] == self.year

        # Count of those in care by age and year
        in_care_age = (self.population.loc[in_care].groupby(['age']).size()
                       .reindex(index=AGES, fill_value=0).reset_index(name='n')
                       .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.in_care_age = self.stats.in_care_age.append(in_care_age)

        # Count of those in care by age and year
        out_care_age = (self.population.loc[out_care].groupby(['age']).size()
                        .reindex(index=AGES, fill_value=0).reset_index(name='n')
                        .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.out_care_age = self.stats.out_care_age.append(out_care_age)

        # Count of those reengaging in care by age and year
        reengaged_age = (self.population.loc[reengaged].groupby(['age']).size()
                         .reindex(index=AGES, fill_value=0).reset_index(name='n')
                         .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.reengaged_age = self.stats.reengaged_age.append(reengaged_age)

        # Count of those lost to care by age and year
        ltfu_age = (self.population.loc[ltfu].groupby(['age']).size()
                    .reindex(index=AGES, fill_value=0).reset_index(name='n')
                    .assign(year=self.year, replication=self.replication, group=self.group_name))
        self.stats.ltfu_age = self.stats.ltfu_age.append(ltfu_age)

        # Discretize cd4 count and count those in care
        cd4_in_care = pd.DataFrame((self.population.loc[in_care, 'time_varying_sqrtcd4n']).round(0).astype(int)).rename(columns={'time_varying_sqrtcd4n': 'cd4_count'})
        cd4_in_care = cd4_in_care.groupby('cd4_count').size().reindex(np.arange(51), fill_value=0)
        cd4_in_care = cd4_in_care.reset_index(name='n').assign(year=self.year, replication=self.replication, group=self.group_name)
        self.stats.cd4_in_care = self.stats.cd4_in_care.append(cd4_in_care)

        # Discretize cd4 count and count those out care
        cd4_out_care = pd.DataFrame((self.population.loc[out_care, 'time_varying_sqrtcd4n']).round(0).astype(int)).rename(columns={'time_varying_sqrtcd4n': 'cd4_count'})
        cd4_out_care = cd4_out_care.groupby('cd4_count').size().reindex(np.arange(51), fill_value=0)
        cd4_out_care = cd4_out_care.reset_index(name='n').assign(year=self.year, replication=self.replication, group=self.group_name)
        self.stats.cd4_out_care = self.stats.cd4_out_care.append(cd4_out_care)

        if self.parameters.comorbidity_flag:
            for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
                has_condition = self.population[condition] == 1

                prevalence_in_care = (self.population.loc[in_care & has_condition].groupby(['age_cat']).size()
                                      .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                      .assign(year=self.year, replication=self.replication, group=self.group_name, condition=condition))
                self.stats.prevalence_in_care = self.stats.prevalence_in_care.append(prevalence_in_care)

                prevalence_out_care = (self.population.loc[out_care & has_condition].groupby(['age_cat']).size()
                                       .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                       .assign(year=self.year, replication=self.replication, group=self.group_name, condition=condition))
                self.stats.prevalence_out_care = self.stats.prevalence_out_care.append(prevalence_out_care)

                prevalence_inits = (self.population.loc[initiating & has_condition].groupby(['age_cat']).size()
                                    .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                    .assign(year=self.year, replication=self.replication, group=self.group_name, condition=condition))
                self.stats.prevalence_inits = self.stats.prevalence_inits.append(prevalence_inits)

                prevalence_dead = (self.population.loc[dying & has_condition].groupby(['age_cat']).size()
                                   .reindex(index=AGE_CATS, fill_value=0).reset_index(name='n')
                                   .assign(year=self.year, replication=self.replication, group=self.group_name, condition=condition))
                self.stats.prevalence_dead = self.stats.prevalence_dead.append(prevalence_dead)

            mm_in_care = (self.population.loc[in_care].groupby(['age_cat', 'mm']).size()
                          .reindex(index=pd.MultiIndex.from_product([AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                          .reset_index(name='n').assign(year=self.year, replication=self.replication, group=self.group_name))
            self.stats.mm_in_care = self.stats.mm_in_care.append(mm_in_care)

            mm_out_care = (self.population.loc[out_care].groupby(['age_cat', 'mm']).size()
                           .reindex(index=pd.MultiIndex.from_product([AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                           .reset_index(name='n').assign(year=self.year, replication=self.replication, group=self.group_name))
            self.stats.mm_out_care = self.stats.mm_out_care.append(mm_out_care)

            mm_inits = (self.population.loc[initiating].groupby(['age_cat', 'mm']).size()
                        .reindex(index=pd.MultiIndex.from_product([AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                        .reset_index(name='n').assign(year=self.year, replication=self.replication, group=self.group_name))
            self.stats.mm_inits = self.stats.mm_inits.append(mm_inits)

            mm_dead = (self.population.loc[dying].groupby(['age_cat', 'mm']).size()
                       .reindex(index=pd.MultiIndex.from_product([AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                       .reset_index(name='n').assign(year=self.year, replication=self.replication, group=self.group_name))
            self.stats.mm_dead = self.stats.mm_dead.append(mm_dead)

            if self.parameters.mm_detail_flag:
                mm_detail_in_care = create_mm_detail_stats(self.population.loc[in_care].copy())
                mm_detail_in_care = mm_detail_in_care.assign(year=self.year, replication=self.replication, group=self.group_name)
                self.stats.mm_detail_in_care = self.stats.mm_detail_in_care.append(mm_detail_in_care)

                mm_detail_out_care = create_mm_detail_stats(self.population.loc[out_care].copy())
                mm_detail_out_care = mm_detail_out_care.assign(year=self.year, replication=self.replication, group=self.group_name)
                self.stats.mm_detail_out_care = self.stats.mm_detail_out_care.append(mm_detail_out_care)

                mm_detail_inits = create_mm_detail_stats(self.population.loc[initiating].copy())
                mm_detail_inits = mm_detail_inits.assign(year=self.year, replication=self.replication, group=self.group_name)
                self.stats.mm_detail_inits = self.stats.mm_detail_inits.append(mm_detail_inits)

                mm_detail_dead = create_mm_detail_stats(self.population.loc[dying].copy())
                mm_detail_dead = mm_detail_dead.assign(year=self.year, replication=self.replication, group=self.group_name)
                self.stats.mm_detail_dead = self.stats.mm_detail_dead.append(mm_detail_dead)

    def record_final_stats(self):
        dead_in_care = self.population['status'] == DEAD_ART_USER
        dead_out_care = self.population['status'] == DEAD_ART_NONUSER
        new_inits = self.population['h1yy'] >= 2010

        # Count of new initiators by year and age
        new_init_age = self.population.loc[new_inits].groupby(['h1yy', 'init_age']).size()
        new_init_age = new_init_age.reindex(pd.MultiIndex.from_product([SIMULATION_YEARS, AGES], names=['year', 'age']), fill_value=0)
        self.stats.new_init_age = new_init_age.reset_index(name='n').assign(replication=self.replication, group=self.group_name)

        # Count of those that died in care by age and year
        dead_in_care_age = self.population.loc[dead_in_care].groupby(['year_died', 'age']).size()
        dead_in_care_age = dead_in_care_age.reindex(pd.MultiIndex.from_product([SIMULATION_YEARS, AGES], names=['year', 'age']), fill_value=0)
        self.stats.dead_in_care_age = dead_in_care_age.reset_index(name='n').assign(replication=self.replication, group=self.group_name)

        # Count of those that died out of care by age and year
        dead_out_care_age = self.population.loc[dead_out_care].groupby(['year_died', 'age']).size()
        dead_out_care_age = dead_out_care_age.reindex(pd.MultiIndex.from_product([SIMULATION_YEARS, AGES], names=['year', 'age']), fill_value=0)
        self.stats.dead_out_care_age = dead_out_care_age.reset_index(name='n').assign(replication=self.replication, group=self.group_name)

        # Count of discretized cd4 count at ART initiation
        cd4_inits = self.population[['init_sqrtcd4n', 'h1yy']].copy()
        cd4_inits['cd4_count'] = (cd4_inits['init_sqrtcd4n']).round(0).astype(int)
        cd4_inits = cd4_inits.groupby(['h1yy', 'cd4_count']).size()
        cd4_inits = cd4_inits.reindex(pd.MultiIndex.from_product([ALL_YEARS, np.arange(51)], names=['year', 'cd4_count']), fill_value=0)
        self.stats.cd4_inits = cd4_inits.reset_index(name='n').assign(replication=self.replication, group=self.group_name)

    def run(self):
        """ Simulate from 2010 to 2030 """
        while self.year <= 2030:

            # Apply smoking intervention
            if (self.year == 2010) & self.parameters.smoking_intervention:
                self.population['smoking'] = 0

            # Everybody ages
            self.increment_age()

            # Apply comorbidities
            if self.parameters.comorbidity_flag:
                self.apply_stage_1()
                self.apply_stage_2_and_3()
                self.update_mm()

            # In care operations
            self.increase_cd4_count()  # Increase cd4n in people in care
            self.add_new_user()  # Add in newly diagnosed ART initiators
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

            if self.parameters.verbose:
                print(self)

            # Increment year
            self.year += 1
        self.record_final_stats()
