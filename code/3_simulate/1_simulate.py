# Imports
from os import getcwd
import ray
import pearl

###############################################################################
# Main Function                                                               #
###############################################################################
@ray.remote
def run(parameters, group_name, replication):
    pearl.Pearl(parameters, group_name, replication, verbose=False)
    return True


ray.init(num_cpus=7)
param_file = getcwd() + '/../../data/parameters/parameters.h5'
replications = range(1)

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

group_names = ['het_hisp_female']

sa_dict = {'lambda1':               0,
           'mu1':                   None,
           'mu2':                   None,
           'sigma1':                None,
           'sigma2':                None,
           'mortality_in_care':     None,
           'mortality_out_care':    None,
           'loss_to_follow_up':     None,
           'cd4_increase':          None,
           'cd4_decrease':          None}


for group_name in group_names:
    print(group_name)
    futures = [run.remote(pearl.Parameters(path=param_file, group_name=group_name, comorbidity_flag=0, dx_reduce_flag=0,
                                           sa_dict=sa_dict), group_name, replication)
               for replication in replications]
    ray.get(futures)
