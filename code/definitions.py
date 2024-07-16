# Django style root dir definition
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = Path(__file__).parent.parent.resolve()
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
