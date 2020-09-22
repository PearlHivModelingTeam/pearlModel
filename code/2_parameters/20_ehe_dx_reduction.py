import os
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 1001)

# Define directories
cwd = os.getcwd()
param_dir = cwd + '/../../data/parameters/aim1'
fig_dir   = os.getcwd() + '/../../out/fig/new_dx_reduce'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

new_dx = pd.read_feather(f'{param_dir}/new_dx_interval.feather')

dx_2020 = new_dx.loc[new_dx['year'] == 2020].copy()
dx_2020['2020'] = (dx_2020['upper'] + dx_2020['lower']) / 2.0
dx_2020['diff'] = (dx_2020['upper'] - dx_2020['lower']) / 2.0
dx_reduce = dx_2020[['group', '2020']].set_index('group')
print(dx_reduce)

dx_reduce['2021'] = dx_reduce['2020'] - 0.1875 * dx_reduce['2020']
dx_reduce['2022'] = dx_reduce['2021'] - 0.1875 * dx_reduce['2020']
dx_reduce['2023'] = dx_reduce['2022'] - 0.1875 * dx_reduce['2020']
dx_reduce['2024'] = dx_reduce['2023'] - 0.1875 * dx_reduce['2020']
dx_reduce['2025'] = dx_reduce['2024']
dx_reduce['2026'] = dx_reduce['2024']
dx_reduce['2027'] = dx_reduce['2024']
dx_reduce['2028'] = dx_reduce['2024']
dx_reduce['2029'] = dx_reduce['2024']
dx_reduce['2030'] = dx_reduce['2024']
dx_reduce = dx_reduce.reset_index()

year_list = [str(year) for year in range(2020, 2031)]

dx_reduce = pd.melt(dx_reduce, id_vars=['group'], value_vars=year_list, var_name='year', value_name='n').sort_values(['group', 'year'])

dx_diff = dx_2020['diff'].to_numpy()
total_years = len(year_list)
dx_reduce['diff'] = np.array([total_years*[diff] for diff in dx_diff]).flatten()

dx_reduce['lower'] = dx_reduce['n'] - dx_reduce['diff']
dx_reduce['upper'] = dx_reduce['n'] + dx_reduce['diff']
dx_reduce = dx_reduce[['group', 'year', 'lower', 'upper']]
dx_reduce['lower'][dx_reduce['lower'] < 0] = 1.0

new_dx_mod = new_dx.loc[new_dx['year'] < 2020]

new_dx_reduce = pd.concat([dx_reduce, new_dx_mod]).sort_values(['group', 'year'])
new_dx_reduce['year'] = new_dx_reduce['year'].astype(int)
new_dx_reduce = new_dx_reduce.reset_index(drop=True)
new_dx_reduce.to_feather(f'{param_dir}/new_dx_combined_ehe.feather')
