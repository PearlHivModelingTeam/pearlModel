# Imports
import os
import numpy as np
import pandas as pd
import feather
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", 1001)

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

# Define directories
cwd = os.getcwd()
in_dir = cwd + '/../../data/input/stage_2'
out_dir = cwd + '/../../data/parameters/stage_2'
