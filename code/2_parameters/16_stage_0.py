# Imports
import os
import pandas as pd
from pathlib import Path

# Define directories
pearl_dir = Path(os.getenv('PEARL_DIR'))
in_dir = f'{pearl_dir}/param/raw/aim2/stage0'
out_dir = f'{pearl_dir}/param/param/aim2/stage0'

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
group_names.sort()

hcv_prev_users = pd.read_csv(f'{in_dir}/hcv_prev_user_by_race_risk_sex.csv')
hcv_prev_users.columns = hcv_prev_users.columns.str.lower()
hcv_prev_users['sex'] = hcv_prev_users['sex'].str.lower()
hcv_prev_users['pop2'] = hcv_prev_users['pop2'].str.lower()
hcv_prev_users['group'] = hcv_prev_users['pop2'] + '_' + hcv_prev_users['sex']
hcv_prev_users['prevalence'] /= 100.0
hcv_prev_users = hcv_prev_users.rename(columns={'prevalence': 'proportion'})[['group', 'proportion']].copy()

hcv_prev_inits = pd.read_csv(f'{in_dir}/hcv_prev_ini_by_race_risk_sex.csv')
hcv_prev_inits.columns = hcv_prev_inits.columns.str.lower()
hcv_prev_inits['sex'] = hcv_prev_inits['sex'].str.lower()
hcv_prev_inits['pop2'] = hcv_prev_inits['pop2'].str.lower()
hcv_prev_inits['group'] = hcv_prev_inits['pop2'] + '_' + hcv_prev_inits['sex']
hcv_prev_inits['prevalence'] /= 100.0
hcv_prev_inits = hcv_prev_inits.rename(columns={'prevalence': 'proportion'})[['group', 'proportion']].copy()


smoking_prev_users = pd.read_csv(f'{in_dir}/smoking_prev_user_by_race_risk_sex.csv')
smoking_prev_users.columns = smoking_prev_users.columns.str.lower()
smoking_prev_users['sex'] = smoking_prev_users['sex'].str.lower()
smoking_prev_users['pop2'] = smoking_prev_users['pop2'].str.lower()
smoking_prev_users['group'] = smoking_prev_users['pop2'] + '_' + smoking_prev_users['sex']
smoking_prev_users['avg'] /= 100.0
smoking_prev_users = smoking_prev_users.rename(columns={'avg': 'proportion'})[['group', 'proportion']].copy()

smoking_prev_inits = pd.read_csv(f'{in_dir}/smoking_prev_ini_by_race_risk_sex.csv')
smoking_prev_inits.columns = smoking_prev_inits.columns.str.lower()
smoking_prev_inits['sex'] = smoking_prev_inits['sex'].str.lower()
smoking_prev_inits['pop2'] = smoking_prev_inits['pop2'].str.lower()
smoking_prev_inits['group'] = smoking_prev_inits['pop2'] + '_' + smoking_prev_inits['sex']
smoking_prev_inits['prevalence'] /= 100.0
smoking_prev_inits = smoking_prev_inits.rename(columns={'prevalence': 'proportion'})[['group', 'proportion']].copy()

hcv_prev_users.to_csv(f'{out_dir}/hcv_prev_users.csv', index=False)
hcv_prev_inits.to_csv(f'{out_dir}/hcv_prev_inits.csv', index=False)
smoking_prev_users.to_csv(f'{out_dir}/smoking_prev_users.csv', index=False)
smoking_prev_inits.to_csv(f'{out_dir}/smoking_prev_inits.csv', index=False)

