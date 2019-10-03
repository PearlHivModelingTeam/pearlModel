# Imports
from os import getcwd
import pearl
import numpy as np
import feather
import matplotlib.pyplot as plt
import scipy.stats as stats

CB = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

param_file = getcwd() + '/../../data/parameters/parameters.h5'
naaccord_2009 = feather.read_dataframe(getcwd() + '/../../data/input/naaccord_2009.feather')
group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
#group_names = ['idu_hisp_female']
def mixture(coeffs, x):
    rv1 = stats.norm(loc=coeffs.loc['mu1', 'estimate'], scale=coeffs.loc['sigma1', 'estimate'])
    rv2 = stats.norm(loc=coeffs.loc['mu2', 'estimate'], scale=coeffs.loc['sigma2', 'estimate'])
    lambda1 = coeffs.loc['lambda1', 'estimate']
    y = lambda1 * rv1.pdf(x) + (1.0 - lambda1) * rv2.pdf(x)
    return (y)

ages = np.arange(18, 85, 1.0)

for group_name in group_names:
    print(group_name)
    parameters = pearl.Parameters(param_file, group_name, False)
    na_grouped = naaccord_2009.loc[naaccord_2009['group'] == group_name]
    plt.hist(na_grouped['age2009'], bins=20, density=True, color=CB[0])
    plt.plot(ages, mixture(parameters.age_in_2009, ages), label='old', color=CB[1])
    plt.plot(ages, mixture(parameters.new_age_in_2009, ages), label='new', color=CB[2])
    plt.legend()
    plt.show()









