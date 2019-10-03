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
replications = 100

data = pd.DataFrame()
for group_name in group_names:
    print(group_name)
    for replication in range(replications):
        print(replication)
        rep_df = feather.read_dataframe(f'{feather_dir}/{group_name}_in_care_count.feather')
        data = data.append(rep_df)


data = data.groupby(['group', 'year', 'age_cat'])['n'].agg(['median', 'std']).reset_index()

feather.write_dataframe(data, f'{rainbow_dir}/sim.feather')





