# Django style root dir definition
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
POPULATION_TYPE_DICT = {
            'age':'int8',
            'age_cat': 'int8',
            'anx' : 'bool',
            'ckd' : 'bool',
            'dm' : 'bool',
            'dpr' : 'bool',
            'esld' : 'bool',
            'h1yy' : 'int16',
            'hcv' : 'bool',
            'ht' : 'bool',
            'init_age' : 'int8',
            'last_h1yy' : 'int16',
            'lipid' : 'bool',
            'ltfu_year' : 'int16',
            'malig' : 'bool',
            'mi' : 'bool',
            'mm' : 'int8',
            'n_lost' : 'int32',
            'return_year' : 'int16',
            'smoking' : 'bool',
            'sqrtcd4n_exit' : 'float64',
            'status' : 'int8',
            't_anx' : 'int16',
            't_ckd' : 'int16',
            't_dm' : 'int16',
            't_dpr' : 'int16',
            't_esld' : 'int16',
            't_hcv' : 'int16',
            't_ht' : 'int16',
            't_lipid' : 'int16',
            't_malig' : 'int16',
            't_mi' : 'int16',
            't_smoking' : 'int16',
            'year' : 'int16',
            'years_out' : 'int16',}

# Status Constants
ART_NAIVE = 0
DELAYED = 1
ART_USER = 2
ART_NONUSER = 3
REENGAGED = 4
LTFU = 5
DYING_ART_USER = 6
DYING_ART_NONUSER = 7
DEAD_ART_USER = 8
DEAD_ART_NONUSER = 9

# Smearing correction
SMEARING = 1.4

# Comorbidity stages
STAGE0 = ['hcv', 'smoking']
STAGE1 = ['anx', 'dpr']
STAGE2 = ['ckd', 'lipid', 'dm', 'ht']
STAGE3 = ['malig', 'esld', 'mi']
