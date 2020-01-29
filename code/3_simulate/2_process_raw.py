# Imports
import os
import pandas as pd
import feather

# print more rows
pd.options.display.max_rows = 6000

# Define directories
cwd = os.getcwd()
h5_dir = cwd + '/../../out/raw'
feather_dir = cwd + '/../../out/processed/new'


group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

#group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male']
#group_names = ['idu_white_female']

replications = 500

for group_name in group_names:
    in_care_count = pd.DataFrame()
    in_care_age = pd.DataFrame()
    out_care_count = pd.DataFrame()
    out_care_age = pd.DataFrame()
    reengaged_count = pd.DataFrame()
    reengaged_age = pd.DataFrame()
    ltfu_count = pd.DataFrame()
    ltfu_age = pd.DataFrame()
    dead_in_care_count = pd.DataFrame()
    dead_in_care_age = pd.DataFrame()
    dead_out_care_count = pd.DataFrame()
    dead_out_care_age = pd.DataFrame()
    new_init_count = pd.DataFrame()
    new_init_age = pd.DataFrame()
    n_unique_out_care = pd.DataFrame()
    years_out = pd.DataFrame()
    random_params = pd.DataFrame()

    #smoking_count = pd.DataFrame()
    #hcv_count = pd.DataFrame()
    #anxiety_count = pd.DataFrame()
    #depression_count = pd.DataFrame()
    #ckd_count = pd.DataFrame()
    #lipid_count = pd.DataFrame()
    #diabetes_count = pd.DataFrame()
    #hypertension_count = pd.DataFrame()

    #dead = pd.DataFrame()
    #smoking_dead = pd.DataFrame()
    #hcv_dead = pd.DataFrame()
    #anxiety_dead = pd.DataFrame()
    #depression_dead = pd.DataFrame()
    #ckd_dead = pd.DataFrame()
    #lipid_dead = pd.DataFrame()
    #diabetes_dead = pd.DataFrame()
    #hypertension_dead = pd.DataFrame()

    #greater_than_zero_count = pd.DataFrame()
    #greater_than_one_count = pd.DataFrame()

    print(group_name)
    for replication in range(replications):
        print(replication)
        with pd.HDFStore(f'{h5_dir}/{group_name}_{replication}.h5') as store:
            in_care_count = in_care_count.append(store['in_care_count'])
            in_care_age = in_care_age.append(store['in_care_age'])

            out_care_count = out_care_count.append(store['out_care_count'])
            out_care_age = out_care_age.append(store['out_care_age'])

            reengaged_count = reengaged_count.append(store['reengaged_count'])
            reengaged_age = reengaged_age.append(store['reengaged_age'])

            ltfu_count = ltfu_count.append(store['ltfu_count'])
            ltfu_age = ltfu_age.append(store['ltfu_age'])

            dead_in_care_count = dead_in_care_count.append(store['dead_in_care_count'])
            dead_in_care_age = dead_in_care_age.append(store['dead_in_care_age'])

            dead_out_care_count = dead_out_care_count.append(store['dead_out_care_count'])
            dead_out_care_age = dead_out_care_age.append(store['dead_out_care_age'])

            new_init_count = new_init_count.append(store['new_init_count'].rename(columns={'h1yy_orig':'h1yy'}))
            new_init_age = new_init_age.append(store['new_init_age'].rename(columns={'h1yy_orig':'h1yy'}))

            n_unique_out_care = n_unique_out_care.append(store['n_unique_out_care'])
            years_out = years_out.append(store['years_out'])

            random_params = random_params.append(store['random_params'])

            """
            smoking_count = smoking_count.append(store['smoking_count'])
            hcv_count = hcv_count.append(store['hcv_count'])
            anxiety_count = anxiety_count.append(store['anxiety_count'])
            depression_count = depression_count.append(store['depression_count'])
            ckd_count = ckd_count.append(store['ckd_count'])
            lipid_count = lipid_count.append(store['lipid_count'])
            diabetes_count = diabetes_count.append(store['diabetes_count'])
            hypertension_count = hypertension_count.append(store['hypertension_count'])

            dead = dead.append(store['dead'])
            smoking_dead = smoking_dead.append(store['smoking_dead'])
            hcv_dead = hcv_dead.append(store['hcv_dead'])
            anxiety_dead = anxiety_dead.append(store['anxiety_dead'])
            depression_dead = depression_dead.append(store['depression_dead'])
            ckd_dead = ckd_dead.append(store['ckd_dead'])
            lipid_dead = lipid_dead.append(store['lipid_dead'])
            diabetes_dead = diabetes_dead.append(store['diabetes_dead'])
            hypertension_dead = hypertension_dead.append(store['hypertension_dead'])

            greater_than_zero_count = greater_than_zero_count.append(store['greater_than_zero_count'])
            greater_than_one_count = greater_than_one_count.append(store['greater_than_one_count'])
            """

    dead_in_care_age = dead_in_care_age.loc[dead_in_care_age.age <= 85]
    dead_out_care_age = dead_out_care_age.loc[dead_out_care_age.age <= 85]

    feather.write_dataframe(in_care_count, f'{feather_dir}/{group_name}_in_care_count.feather')
    feather.write_dataframe(in_care_age, f'{feather_dir}/{group_name}_in_care_age.feather')
    
    feather.write_dataframe(out_care_count, f'{feather_dir}/{group_name}_out_care_count.feather')
    feather.write_dataframe(out_care_age, f'{feather_dir}/{group_name}_out_care_age.feather')

    feather.write_dataframe(reengaged_count, f'{feather_dir}/{group_name}_reengaged_count.feather')
    feather.write_dataframe(reengaged_age, f'{feather_dir}/{group_name}_reengaged_age.feather')

    feather.write_dataframe(ltfu_count, f'{feather_dir}/{group_name}_ltfu_count.feather')
    feather.write_dataframe(ltfu_age, f'{feather_dir}/{group_name}_ltfu_age.feather')

    feather.write_dataframe(dead_in_care_count, f'{feather_dir}/{group_name}_dead_in_care_count.feather')
    feather.write_dataframe(dead_in_care_age, f'{feather_dir}/{group_name}_dead_in_care_age.feather')

    feather.write_dataframe(dead_out_care_count, f'{feather_dir}/{group_name}_dead_out_care_count.feather')
    feather.write_dataframe(dead_out_care_age, f'{feather_dir}/{group_name}_dead_out_care_age.feather')

    feather.write_dataframe(new_init_count, f'{feather_dir}/{group_name}_new_init_count.feather')
    feather.write_dataframe(new_init_age, f'{feather_dir}/{group_name}_new_init_age.feather')

    feather.write_dataframe(n_unique_out_care, f'{feather_dir}/{group_name}_n_unique_out_care.feather')
    feather.write_dataframe(years_out, f'{feather_dir}/{group_name}_years_out.feather')

    feather.write_dataframe(random_params, f'{feather_dir}/{group_name}_random_params.feather')

    """
    feather.write_dataframe(smoking_count, f'{feather_dir}/{group_name}_smoking_count.feather')
    feather.write_dataframe(hcv_count, f'{feather_dir}/{group_name}_hcv_count.feather')
    feather.write_dataframe(anxiety_count, f'{feather_dir}/{group_name}_anxiety_count.feather')
    feather.write_dataframe(depression_count, f'{feather_dir}/{group_name}_depression_count.feather')
    feather.write_dataframe(ckd_count, f'{feather_dir}/{group_name}_ckd_count.feather')
    feather.write_dataframe(lipid_count, f'{feather_dir}/{group_name}_lipid_count.feather')
    feather.write_dataframe(diabetes_count, f'{feather_dir}/{group_name}_diabetes_count.feather')
    feather.write_dataframe(hypertension_count, f'{feather_dir}/{group_name}_hypertension_count.feather')

    feather.write_dataframe(dead, f'{feather_dir}/{group_name}_dead.feather')
    feather.write_dataframe(smoking_dead, f'{feather_dir}/{group_name}_smoking_dead.feather')
    feather.write_dataframe(hcv_dead, f'{feather_dir}/{group_name}_hcv_dead.feather')
    feather.write_dataframe(anxiety_dead, f'{feather_dir}/{group_name}_anxiety_dead.feather')
    feather.write_dataframe(depression_dead, f'{feather_dir}/{group_name}_depression_dead.feather')
    feather.write_dataframe(ckd_dead, f'{feather_dir}/{group_name}_ckd_dead.feather')
    feather.write_dataframe(lipid_dead, f'{feather_dir}/{group_name}_lipid_dead.feather')
    feather.write_dataframe(diabetes_dead, f'{feather_dir}/{group_name}_diabetes_dead.feather')
    feather.write_dataframe(hypertension_dead, f'{feather_dir}/{group_name}_hypertension_dead.feather')

    feather.write_dataframe(greater_than_zero_count, f'{feather_dir}/{group_name}_greater_than_zero_count.feather')
    feather.write_dataframe(greater_than_one_count, f'{feather_dir}/{group_name}_greater_than_one_count.feather')
    """

    


