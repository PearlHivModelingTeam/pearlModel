# Imports
suppressMessages(library(tidyverse))

# Define directories
cwd = getwd()
param_dir = paste0(cwd, '/../../data/param')

# Read in cd4 increase coefficients
cd4_increase_param <- get(load(paste0(param_dir, '/coeff_cd4_increase_190508.rda')))
coeff_cd4_increase <- cd4_increase_param %>% select(-c('vcov', 'model')) %>% unnest

# Read in mortality in care coefficients
coeff_mortality_in <- get(load(paste0(param_dir, '/coeff_mortality_in_care_190508.rda')))
coeff_mortality_in <- coeff_mortality_in %>% select(-c('vcov', 'model')) %>% unnest
