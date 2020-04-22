# Imports
import os
import pandas as pd
import feather

# print more rows
pd.options.display.max_rows = 6000

# Define directories
cwd = os.getcwd()
h5_dir = cwd + '/../../out/raw'
feather_dir = cwd + '/../../out/processed/4_15'

comorbidity = False

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']

#group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male']
#group_names = ['msm_black_male']

replications = 200

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
    initial_cd4n = pd.DataFrame()
    art_coeffs = pd.DataFrame()
    median_cd4s = pd.DataFrame()
    tv_cd4_2009 = pd.DataFrame()

    multimorbidity_in_care = pd.DataFrame()
    multimorbidity_dead = pd.DataFrame()

    anxiety_incidence = pd.DataFrame()
    depression_incidence = pd.DataFrame()
    ckd_incidence = pd.DataFrame()
    lipid_incidence = pd.DataFrame()
    diabetes_incidence = pd.DataFrame()
    hypertension_incidence = pd.DataFrame()

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

            initial_cd4n = initial_cd4n.append(store['initial_cd4n'])

            art_coeffs = art_coeffs.append(store['art_coeffs'])
            median_cd4s = median_cd4s.append(store['median_cd4s'])

            tv_cd4_2009 = tv_cd4_2009.append(store['tv_cd4_2009'])

            if comorbidity:
                multimorbidity_in_care = multimorbidity_in_care.append(store['multimorbidity_in_care'])
                multimorbidity_dead = multimorbidity_dead.append(store['multimorbidity_dead'])

                anxiety_incidence = anxiety_incidence.append(store['anxiety_incidence'])
                depression_incidence = depression_incidence.append(store['depression_incidence'])
                ckd_incidence = ckd_incidence.append(store['ckd_incidence'])
                lipid_incidence = lipid_incidence.append(store['lipid_incidence'])
                diabetes_incidence = diabetes_incidence.append(store['diabetes_incidence'])
                hypertension_incidence = hypertension_incidence.append(store['hypertension_incidence'])

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

    feather.write_dataframe(initial_cd4n, f'{feather_dir}/{group_name}_initial_cd4n.feather')

    feather.write_dataframe(art_coeffs, f'{feather_dir}/{group_name}_art_coeffs.feather')
    feather.write_dataframe(median_cd4s, f'{feather_dir}/{group_name}_median_cd4s.feather')

    feather.write_dataframe(tv_cd4_2009, f'{feather_dir}/{group_name}_tv_cd4_2009.feather')

    if comorbidity:

        feather.write_dataframe(multimorbidity_in_care, f'{feather_dir}/{group_name}_multimorbidity_in_care.feather')
        feather.write_dataframe(multimorbidity_dead, f'{feather_dir}/{group_name}_multimorbidity_dead.feather')

        feather.write_dataframe(anxiety_incidence, f'{feather_dir}/{group_name}_anxiety_incidence.feather')
        feather.write_dataframe(depression_incidence, f'{feather_dir}/{group_name}_depression_incidence.feather')
        feather.write_dataframe(ckd_incidence, f'{feather_dir}/{group_name}_ckd_incidence.feather')
        feather.write_dataframe(lipid_incidence, f'{feather_dir}/{group_name}_lipid_incidence.feather')
        feather.write_dataframe(diabetes_incidence, f'{feather_dir}/{group_name}_diabetes_incidence.feather')
        feather.write_dataframe(hypertension_incidence, f'{feather_dir}/{group_name}_hypertension_incidence.feather')
