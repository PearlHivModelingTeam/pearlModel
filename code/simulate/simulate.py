
# Imports
import sys
import os
import numpy as np
import pandas as pd
import ray
import scipy.stats as stats
from collections import namedtuple
from pearl import Pearl 

# Define directories
cwd = os.getcwd()
proc_dir = cwd + '/../../data/processed'
out_dir = cwd + '/../../out'

# Load everything
with pd.HDFStore(proc_dir + '/converted.h5') as store:
    on_art_2009 = store['on_art_2009']
    mixture_2009_coeff = store['mixture_2009_coeff']
    naaccord_prop_2009 = store['naaccord_prop_2009']
    init_sqrtcd4n_coeff_2009 = store['init_sqrtcd4n_coeff_2009']
    new_dx = store['new_dx']
    new_dx_interval = store['new_dx_interval']
    mixture_h1yy_coeff= store['mixture_h1yy_coeff']
    init_sqrtcd4n_coeff = store['init_sqrtcd4n_coeff']
    cd4_increase_coeff = store['cd4_increase_coeff']
    cd4_decrease_coeff = store['cd4_decrease_coeff']
    ltfu_coeff = store['ltfu_coeff']
    mortality_in_care_coeff = store['mortality_in_care_coeff']
    mortality_out_care_coeff = store['mortality_out_care_coeff']
    prob_reengage = store['prob_reengage']

###############################################################################
# Main Function                                                               #
###############################################################################

def main():
    group_name = 'idu_hisp_female'
    print(group_name)
    replications = 20
    for replication in range(replications):
        print(replication)
        pearl = Pearl(group_name, replication, False)
        pearl.run_simulation(end=2030)

if (__name__ == '__main__'):
    main()
