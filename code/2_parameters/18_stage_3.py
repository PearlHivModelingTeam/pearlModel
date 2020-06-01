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
in_dir = cwd + '/../../data/input/aim2/stage3'
out_dir = cwd + '/../../data/parameters/aim2/stage3'

# Load ckd coeff csv file
df = pd.read_csv(f'{in_dir}/ckd_coeff.csv')

df.columns = df.columns.str.lower()
df = df[['pop2_', 'sex', 'parm', 'estimate']]
df['sex'] = np.where(df['sex'] ==1, 'male', 'female')
df['pop2_'] = df['pop2_'].str.lower()
df['parm'] = df['parm'].str.lower()
