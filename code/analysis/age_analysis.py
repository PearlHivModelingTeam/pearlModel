# Imports
import sys
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# print more rows
pd.options.display.max_rows = 150

# Define directories
cwd = os.getcwd()
out_dir = cwd + '/../../out'


with pd.HDFStore(out_dir + '/out.h5') as store:
    age_final = store['age_final']

x = age_final.loc['msm_black_male'].reset_index()
means = x.groupby(['age_cat', 'year']).mean().drop(columns='dataset').rename(columns={'n': 'mean'})
#print(means)
total = means.reset_index().sort_values('year').set_index(['year', 'age_cat'])
total = total.groupby('year').sum().rename(columns={'mean':'total'})
print(total)

plt.plot(means.loc[2.0].index.values, means.loc[2.0,'mean']/total.total)
plt.ylim(0, 0.4)
plt.show()



