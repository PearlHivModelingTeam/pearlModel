# Imports
library(tidyverse)

# Define directories
cwd = getwd()
in_dir = '/home/cameron/silvertsunami/statepilsna18/lhume/naaccord/Silver/Data'
out_dir = paste0(cwd, '/../data/processed')

# Read in, get coeffs, save cd4 increase data
cd4_increase_param <- get(load(paste0(in_dir, '/parameters/coeff_cd4_increase_190508.Rda')))
coeff_cd4_increase <- cd4_increase_param %>% select(-c('vcov', 'model')) %>% unnest
save(coeff_cd4_increase, file = paste0(out_dir, '/coeff_cd4_increase.rda'))

# Load preprocessed data, and save dx_interval 
load(paste0(out_dir,"/sense.data"))
dx_interval <- test %>% select(c('group', 'sex', 'hiv_pred_interval')) %>% unnest
save(dx_interval, file = paste0(out_dir, '/dx_interval.rda'))

