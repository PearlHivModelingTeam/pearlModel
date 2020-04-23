# Imports
import os
import shutil
import ray
import pearl

###############################################################################
# Main Function                                                               #
###############################################################################
@ray.remote
def run(parameters, group_name, replication):
    pearl.Pearl(parameters, group_name, replication, verbose=False)
    return True

# Number of cores
ray.init(num_cpus=7)

# Input and output files
param_file = f'{os.getcwd()}/../../data/parameters/parameters.h5'
output_folder = f'{os.getcwd()}/../../out/test/test'

# Number of replications
replications = range(1)

# Groups to run
group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male']
group_names = ['het_hisp_female']

# Declare sensitivity analysis params
sa_dict = {'lambda1':               None,
           'mu1':                   None,
           'mu2':                   None,
           'sigma1':                None,
           'sigma2':                None,
           'mortality_in_care':     None,
           'mortality_out_care':    None,
           'loss_to_follow_up':     None,
           'cd4_increase':          None,
           'cd4_decrease':          None,
           'new_pop_size':          None}

# Delete old files
if os.path.isdir(output_folder):
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Run simulations
for group_name in group_names:
    print(group_name)
    futures = [run.remote(pearl.Parameters(path=param_file, group_name=group_name, comorbidity_flag=0, dx_reduce_flag=0,
                                           sa_dict=sa_dict, output_folder=output_folder), group_name, replication)
               for replication in replications]
    ray.get(futures)
