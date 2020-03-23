import os
import feather
import pandas as pd
import statsmodels.stats.weightstats as weightstats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../../out/processed/new_reengage'
param_dir = cwd + '/../../data/parameters/aim1'
out_dir = cwd + '/out'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

data = pd.concat([pd.read_feather(f'{in_dir}/{group_name}_years_out.feather') for group_name in group_names])
print(data)