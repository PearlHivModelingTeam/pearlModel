# Imports
from os import getcwd 
import pandas as pd
import ray
import pearl

###############################################################################
# Main Function                                                               #
###############################################################################
@ray.remote
def run(parameters, group_name, replication):
    pearl.Pearl(parameters, group_name, replication, False, False)
    return True

ray.init(num_cpus=7)
proc_dir = getcwd() + '/../../data/processed'
replications = 1000

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
for group_name in group_names:
    print(group_name)
    parameters = pearl.Parameters(proc_dir, group_name)
    futures = [run.remote(parameters, group_name, replication) for replication in range(replications)]
    ray.get(futures)

