# Imports
import os
import numpy as np
import pandas as pd
import itertools

# Stat modeling packages
from statsmodels.gam.api import GLMGam, BSplines
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.mixture as skm

# R to python interface
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate R interface
base = importr('base')
pandas2ri.activate()

# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../../data/input'
param_dir = cwd + '/../../data/param'
out_dir = cwd + '/../../data/processed'

###############################################################################
# Functions                                                                   #
###############################################################################

def expand_grid(data_dict):
    """Replicate expand.grid() from R"""

    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def test_group(df):
    """Filter to a single group for testing"""
    df = df.loc[df['group'] == 'het_black_female']
    return df

def get_on_art(df):
    """Format CDC surveillance data and get number of people on art in 2009"""
    df.columns = df.columns.str.lower()

    # Make combined 'group' index
    df.sex = np.where(df.sex == 'Males', 'male', 'female')
    df.group = df.group + '_' + df.sex

    # Calculate number of people on ART
    df['on_art'] = np.floor(0.01 * df.n_alive_2009 * df.pct_art)
    
    # Drop extra columns and make 'group' the index
    df = df.drop(['sex','surveillance report', 'table', 'n_alive_2009', 'pct_art', 'pct_source', 'notes'], axis=1)
    df = df.sort_values(by='group')
    df = df.set_index('group')

    return(df)

def format_naaccord(df):
    """Format naaccord population data"""
    
    # Replace ART init date with year
    df['art_init'] = pd.DatetimeIndex(df.haart1date).year
    
    # Format column names and sex values
    df = df.rename(columns={'pop2':'group'})
    df.group = df.group.str.decode('utf-8')
    df.columns = df.columns.str.lower()
    df.group = df.group.str.lower()
    
    # Drop rows with missing group and sex
    df.loc[df['sex'] == 3, 'sex'] = np.nan 
    df = df.dropna(subset=['group','sex'])
    
    # Replace int with string for sex
    df['sex'] = np.where(df['sex'] == 1, 'male', 'female')
   
    # Convert from SAS dates
    df.obs_entry = pd.to_timedelta(df.obs_entry, unit='D') + pd.Timestamp('1960-1-1')
    df.obs_exit = pd.to_timedelta(df.obs_exit, unit='D') + pd.Timestamp('1960-1-1')

    # Replace negative cd4n with NaN
    df.loc[df['cd4n'] < 0 , 'cd4n'] = np.nan

    # Recode study exit to be based on min(death date, cohort closedate)
    df.obs_exit = np.where((df.deathdate < df.cohort_closedate), df.deathdate, df.cohort_closedate)

    # Set deathdate to missing if reason for exit is not death
    df.deathdate = df.deathdate.where(df.deathdate == df.obs_exit, pd.NaT)

    # Drop unneeded columns and combine group and sex
    df.group = df.group + '_' + df.sex
    df = df.sort_values(by=['group','naid'])
    df = df.drop(['pop1','risk','race','cohort','lab9date','cohort_closedate', 'sex', 'haart1date'], axis=1)

    return(df)

def get_naaccord_2009(df):
    """Get naaccord population that was in care in 2009"""
    df['age2009'] = 2009 - df.yob

    # Read in care status by year for naaccord population
    care_status = pd.read_sas(in_dir + '/popu16_carestatus.sas7bdat')
    care_status.columns = care_status.columns.str.lower()
    care_status = care_status.loc[care_status['year'] == 2009]
    care_status = care_status.loc[care_status['in_care'] == 1]
   
    # Filter naaccord patients to those in care in 2009
    df = pd.merge(df, care_status, on='naid')
    df = df.drop(['year','in_care'], axis=1)

    return df

def get_h1yy_proportions(df):
    """Splits the naaccord population alive in 2009, splits it into age categories and returns the
    number and proportion of people starting art in a certain year"""

    # Define age groups
    df['age2009cat'] = np.floor(df.age2009 / 10)
    df.loc[df.age2009cat > 7, 'age2009cat'] = 7

    #Count how many times art_init occurs for each group and age category
    sum_df = df.groupby(['group','age2009cat']).agg({'naid': 'count'})
    count_df = df.groupby(['group','age2009cat','art_init']).agg({'naid': 'count'})
    count_df['pct'] = count_df.naid / sum_df.naid
    count_df = count_df.rename(columns = {'naid' : 'n'})
    
    return(count_df) 

def get_initial_cd4n(df, result_df):
    """Fit mean and standard deviation of sqrt(cd4n) as a function of art init year with a GLM for each group"""

    # filter to ART inits between 2000 and 2009
    df = df.loc[(df.art_init >= 2000) & (df.art_init <= 2009)].copy()

    # Create sqrt cd4n column and find mean and std deviation per h1yy
    df['sqrtcd4n'] = np.sqrt(df.cd4n)
    dfmean = df.groupby(['group', 'art_init']).mean()[['sqrtcd4n']]
    dfmean = dfmean.rename(columns={'sqrtcd4n': 'mean'})
    dfstd = df.groupby(['group', 'art_init']).std()[['sqrtcd4n']]
    dfstd = dfstd.rename(columns={'sqrtcd4n': 'std'})
    df1 = pd.concat([dfmean, dfstd], axis=1)
    
    # Fit mean and standard deviation of sqrtcd4n ~ art_init for each group
    for name, group in df1.reset_index().groupby('group'):
        model = smf.glm('mean ~ art_init', group)
        results = model.fit()
        result_df.loc[name, 'meanint'] = results.params[0]
        result_df.loc[name, 'meanslope'] = results.params[1]
        model = smf.glm('std ~ art_init', group)
        results = model.fit()
        result_df.loc[name, 'stdint'] = results.params[0]
        result_df.loc[name, 'stdslope'] = results.params[1]
    
    return(result_df)

def format_new_dx(df):
    """Format DataFrame of new dxs by group and year (2009-2015) from CDC Surveillance data"""
        
    # Format race, sex, risk group and column names
    df.columns = df.columns.str.lower()
    df = df.loc[df.race != 'Other']
    df = df.loc[df.risk != 'other']
    df.loc[df.race == 'White Non-Hispanic', 'race'] = 'white' 
    df.loc[df.race == 'Black Non-Hispanic', 'race'] = 'black' 
    df.loc[df.race == 'Hispanic', 'race'] = 'hisp' 
    df.sex = np.where(df.sex == 'Males', 'male', 'female')
    df['group'] = df.risk + '_' + df.race + '_' + df.sex
    df = df.drop(['sex','risk','race'], axis=1)

    # group by group and year and then sum over age category
    df = df.groupby(['group','year']).sum()
    df = df.rename(columns={'n_hivdx_cdctable1' : 'n_dx'})
    df.n_dx = np.floor(df.n_dx)
    
    return(df)

def art_init_fit(df):
    """Fit a mixed normal distribution to age at art initiation for each year, 2009-2015"""
    
    # Create DataFrame to store output
    output = expand_grid({'group': df['group'].unique(),
                                   'art_init': np.arange(2009,2016)})
    output = output.reindex(columns= output.columns.tolist() + ['mu1','mu2','var1','var2','weight1','weight2'])
    output = output.set_index(['group','art_init'])

    # Create column: age at ART init and filter to ART init within 2009-2015
    init_pop = (df.assign(init_age = df.art_init - df.yob)
                  .query('art_init <= 2015 & art_init >= 2009')).copy()

    # Collapse idu female groups
    init_pop.loc[((init_pop.group == "idu_black_female") | (init_pop.group == "idu_white_female")) & (init_pop.art_init.isin([2009, 2010])), 'art_init'] = 2009
    init_pop.loc[((init_pop.group == "idu_black_female") | (init_pop.group == "idu_white_female")) & (init_pop.art_init.isin([2011, 2012])), 'art_init'] = 2011
    init_pop.loc[((init_pop.group == "idu_black_female") | (init_pop.group == "idu_white_female")) & (init_pop.art_init.isin([2013, 2014, 2015])), 'art_init'] = 2013

    # Collapse idu hisp groups
    init_pop.loc[(init_pop.group == "idu_hisp_male") & (init_pop.art_init.isin([2009, 2010])), 'art_init'] = 2009
    init_pop.loc[(init_pop.group == "idu_hisp_male") & (init_pop.art_init.isin([2011, 2012])), 'art_init'] = 2011
    init_pop.loc[(init_pop.group == "idu_hisp_male") & (init_pop.art_init.isin([2013, 2014, 2015])), 'art_init'] = 2013
    init_pop.loc[(init_pop.group == "idu_hisp_female"), 'art_init'] = 2009
    
    # For each ART init year, fit age to a two component mixed gaussian and store in output DataFrame
    for name, group in init_pop.groupby(['group', 'art_init']):
        gm = skm.GaussianMixture(n_components=2, means_init = [[25],[50]], max_iter=4000, tol=1e-8, covariance_type='spherical').fit(group.init_age.values.reshape(-1,1))
        output.loc[name,'mu1'] = gm.means_[0][0]
        output.loc[name,'mu2'] = gm.means_[1][0]
        output.loc[name,'var1'] = gm.covariances_[0]
        output.loc[name,'var2'] = gm.covariances_[1]
        output.loc[name,'weight1'] = gm.weights_[0]
        output.loc[name,'weight2'] = gm.weights_[1]
    
    # Fill missing values with previous year's value and return
    output = output.fillna(method='ffill')
    return(output)

def gmix_param_fit(df):
    """ Fit a GLM to each parameter from the Gaussian mixture model and predict parameters into 2030"""
    df = df.reset_index()
    
    # Create DataFrame to store output
    output = expand_grid({'group': df.group.unique(),
                                   'art_init': np.arange(2009,2031)})
    output = output.reindex(columns = output.columns.tolist() + ['mu1','mu2','var1','var2','weight1','weight2'])
    output = output.set_index(['group'])

    for name, group in output.reset_index().groupby('group'):
        group_data = df.loc[df.group == name]
        if (name != 'idu_hisp_female'):
            for column in group_data:
                if ((column != 'group') & (column != 'art_init')):
                    model = smf.glm(column + ' ~ art_init', group_data)
                    result = model.fit()
                    pred = result.get_prediction(group)
                    output.loc[name, column] = pred.summary_frame()['mean'].values
        else:
            for column in output:
                pass
                # idu_hisp_female is just constant (statsmodels was throwing an error)
                output.loc['idu_hisp_female', column] = group_data[column].iloc[0]

    return(output.reset_index().set_index(['group','art_init']))

def format_age_2009_ci(df):
    df.sex = np.where(df.sex == 'Males', 'male', 'female')
    df.group = df.group + '_' + df.sex
    df = df.drop('sex', axis=1)
    df = df.set_index('group')
    return(df)

def coeff_format(df):
    """Format tables holding coefficient values"""
    df.columns = df.columns.str.lower() 
    
    # Combine sex risk and race into single group indentifier
    df['sex'] = np.where(df['sex'] == 1, 'male', 'female')
    if ('pop2' in df.columns): #Hacking
        df = df.rename(columns={'pop2':'group'})
        df.group = df.group.str.decode('utf-8')
    else:
        df.group = df.group.astype('str')

    df.group = df.group + '_' + df.sex
    df.group = df.group.str.lower()
    df = df.drop(columns='sex')
    df = df.sort_values(by='group')
    df = df.set_index('group')
    return(df)

###############################################################################
# Main Function                                                               #
###############################################################################
def main():
    """Perform preproccessing for the PEARL model""" 

    # Read in and format cdc surveillance data
    cdc_surv_2009 = pd.read_csv(in_dir + '/surveillance_estimates_cdc_2009.csv')

    # Get number of people on art in 2009
    on_art = get_on_art(cdc_surv_2009)

    # Create template for saving coefficients
    template = on_art.drop(['on_art'], axis=1)

    # Read in and format naaccord data
    naaccord = format_naaccord(pd.read_sas(in_dir + '/popu16.sas7bdat'))

    # Get population of NA-ACCORD participants in care in 2009
    naaccord_2009 = get_naaccord_2009(naaccord.copy())

    # Get proportion of 2009 population starting art in a given year grouped by age
    naaccord_prop_2009 = get_h1yy_proportions(naaccord_2009.copy())

    # Fit mean and stddev of initial sqrt(cd4n) as a glm of art init year for each group in 2009 pop
    init_cd4n_coeffs = get_initial_cd4n(naaccord_2009.copy(), template.copy())

    # Fit a mixed normal distribution to age at art initiation for each year, 2009-2015
    init_age_gmix_coeffs = art_init_fit(naaccord.copy())

    # Fit a GLM to each parameter from the Gaussian mixture model and predict parameters into 2030
    gmix_param_coeffs = gmix_param_fit(init_age_gmix_coeffs)

    # Read in CD4 decrease coefficients 
    coeff_cd4_decrease = pd.read_sas(param_dir + '/coeff_cd4_decrease_190508.sas7bdat')
    coeff_cd4_decrease.columns = coeff_cd4_decrease.columns.str.lower()

    # Read in out-of-care mortality coefficients 
    coeff_mortality_out = coeff_format(pd.read_sas(param_dir + '/coeff_mortality_out_care_190508.sas7bdat'))

    # Read in loss to follow up coefficients 
    coeff_ltfu = coeff_format(pd.read_sas(param_dir + '/coeff_ltfu_190508.sas7bdat'))

    # Percentiles used for spline in LTFU model
    pctls_ltfu = coeff_format(pd.read_sas(param_dir + '/pctls_ltfu_190508.sas7bdat'))
    
    # Gaussian mixture coefficient confidence intervals for age in 2009
    base.load(param_dir + '/age2009_mixture_ci.rda') 
    coeff_age_2009_ci = format_age_2009_ci(robjects.r['age2009_mixture_ci'])
   
    # Convert some r objects
    robjects.r.source('convert.r')

    # CD4 increase coefficients
    coeff_cd4_increase = coeff_format(robjects.r['coeff_cd4_increase'])

    # Mortality in care coefficients
    coeff_mortality_in = coeff_format(robjects.r['coeff_mortality_in'])

    # Predicted interval of new diagnoses 
    base.load(param_dir + '/dx_interval.rda') 
    dx_interval = coeff_format(robjects.r['dx_interval'])
    dx_interval = dx_interval.reset_index().set_index(['group','year'])
    
    
    # Save everything
    with pd.HDFStore(out_dir + '/preprocessed.h5') as store:
        store['on_art'] = on_art
        store['naaccord'] = naaccord
        store['naaccord_2009'] = naaccord_2009
        store['naaccord_prop_2009'] = naaccord_prop_2009
        store['init_cd4n_coeffs'] = init_cd4n_coeffs
        store['dx_interval'] = dx_interval
        store['init_age_gmix_coeffs'] = init_age_gmix_coeffs
        store['gmix_param_coeffs'] = gmix_param_coeffs
        store['coeff_age_2009_ci'] = coeff_age_2009_ci
        store['coeff_cd4_decrease'] = coeff_cd4_decrease
        store['coeff_cd4_increase'] = coeff_cd4_increase
        store['coeff_mortality_out'] = coeff_mortality_out
        store['coeff_mortality_in'] = coeff_mortality_in
        store['coeff_ltfu'] = coeff_ltfu
        store['pctls_ltfu'] = pctls_ltfu

main()
