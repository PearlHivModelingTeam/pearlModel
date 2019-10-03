import os
import feather
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
pd.set_option("display.max_rows", 1001)

rainbow_dir = f'{os.getcwd()}/../../out/rainbow_plots'
feather_dir = f'{os.getcwd()}/../../out/feather/py_reset'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'het_white_male', 'idu_hisp_female',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

sim = feather.read_dataframe(f'{rainbow_dir}/sim.feather')
sim = sim.rename(columns={'year': 'calyy', 'age_cat': 'agecat', 'median': 'n'}).set_index(['group', 'calyy'])
N = pd.DataFrame(sim.groupby(['group', 'calyy'])['n'].sum()).rename(columns={'n': 'N'})
sim['N'] = N
sim['median_pct'] = 100 * sim['n'] / sim['N']
sim['p25_pct'] = sim['median_pct'] - 100 * 0.675 * sim['std'] / sim['N']
sim['p75_pct'] = sim['median_pct'] + 100 * 0.675 * sim['std'] / sim['N']

feather.write_dataframe(sim.reset_index(), f'{rainbow_dir}/sim2.feather')
print(sim.loc['idu_hisp_female'])
