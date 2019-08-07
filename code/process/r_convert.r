######################################################################################
# Load packages
######################################################################################

#suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(haven))
#suppressPackageStartupMessages(library(lubridate))
#suppressPackageStartupMessages(library(broom))
#suppressPackageStartupMessages(library(geepack))
#suppressPackageStartupMessages(library(mixtools))
#suppressPackageStartupMessages(library(RGeode))
#suppressPackageStartupMessages(library(triangle))
#suppressPackageStartupMessages(library(gamlss))
#suppressPackageStartupMessages(library(binom))
suppressPackageStartupMessages(library(tidyverse))

wd <- getwd()

paramwd <- paste0(wd, "/../../data/param")
naaccordwd <- paste0(wd, "/../../data/input")
procwd <- paste0(wd, "/../../data/processed")

######################################################################################
# Call function source file
######################################################################################
source(paste0(wd,"/fx.r"))
options(dplyr.print_max = 100)

# Load Functions
indir <- paste0(wd,"/../../data/processed")
load(paste0(procwd,"/processed.rda"))

test <- test %>% mutate(sex = replace(sex, sex == 'Males', 'male'),
                        sex = replace(sex, sex == 'Females', 'female')) %>%
                 unite(group, c('group', 'sex'), remove=TRUE) %>% 
                 arrange(group)

# on_art
on_art <- test %>% select(group, on_art)

# mixture_2009
mixture_2009 <- test %>% select(group, mixture_2009) %>% unnest

# naaccord
naaccord <- test %>% select(group, data_popu) %>% unnest

# naaccord_prop_2009
naaccord_prop_2009 <- test %>% select(group, naaccord_prop_2009) %>% unnest

# init_sqrtcd4n_coeffs
init_sqrtcd4n_coeffs <- test %>% select(group, naaccord_cd4_2009) %>% unnest

# coeff_age_2009_ci
#coeff_age_2009_ci <- test %>% select(group, age2009_ci) %>% unnest

# new_dx
new_dx <- test %>% select(group, surv) %>% unnest

# new_dx_interval
new_dx_interval <- test %>% select(group, hiv_pred_interval) %>% unnest

# gmix_param_coeffs
gmix_param_coeffs <- test%>% select(group, ini2) %>% unnest %>% spread(param, pred)

#setwd(paramwd)

## CD4 increase
#cd4_increase <- get(load("coeff_cd4_increase_190508.rda")) %>% 
#        mutate(sex = replace(sex, sex == 'Males', 'male'),
#               sex = replace(sex, sex == 'Females', 'female')) %>%
#        unite(group, c('group', 'sex'), remove=TRUE) %>% 
#        arrange(group)
#
#
#cd4_increase_coeff <- cd4_increase %>% select(group, coeffs) %>% unnest
#cd4_increase_vcov  <- cd4_increase %>% select(group, vcov) %>% mutate(vcov = map(vcov, as_data_frame)) %>% unnest
#
## CD4 decrease
#cd4_decrease <- get(load("coeff_cd4_decrease_190508.rda"))
#cd4_decrease_coeff = cd4_decrease$coeff
#cd4_decrease_vcov = cd4_decrease$vcov 
#
#
## Mortality for persons out of care
#mortality_out <- get(load("coeff_mortality_out_care_190508.rda")) %>%
#        mutate(sex = replace(sex, sex == 'Males', 'male'),
#               sex = replace(sex, sex == 'Females', 'female')) %>%
#        unite(group, c('group', 'sex'), remove=TRUE) %>% 
#        arrange(group)
#
#mortality_out_coeff <- mortality_out %>% select(group, coeffs) %>% unnest
#mortality_out_vcov  <- mortality_out %>% select(group, vcov) %>% 
#                                         mutate(vcov = map(vcov, as_data_frame)) %>%
#                                         unnest
#                                     
## Mortality for persons in care
#mortality_in <- get(load("coeff_mortality_in_care_190508.rda")) %>%
#        mutate(sex = replace(sex, sex == 'Males', 'male'),
#               sex = replace(sex, sex == 'Females', 'female')) %>%
#        unite(group, c('group', 'sex'), remove=TRUE) %>% 
#        arrange(group)
#
#mortality_in_coeff <- mortality_in %>% select(group, coeffs) %>% unnest
#mortality_in_vcov  <- mortality_in %>% select(group, vcov) %>% 
#                                         mutate(vcov = map(vcov, as_data_frame)) %>%
#                                         unnest
#
## Probability of leaving NA-ACCORD
#coeff_leave_na <- get(load("coeff_ltfu_190508.rda")) %>%
#        mutate(sex = replace(sex, sex == 'Males', 'male'),
#               sex = replace(sex, sex == 'Females', 'female')) %>%
#        unite(group, c('group', 'sex'), remove=TRUE) %>% 
#        arrange(group)
#
#leave_na_coeff <- coeff_leave_na %>% select(group, coeffs) %>% unnest
#leave_na_vcov <- coeff_leave_na  %>% select(group, vcov) %>% 
#                                     mutate(vcov = map(vcov, as_data_frame)) %>%
#                                     unnest
#
## Percentiles used for age for the spline in the LTFU model
#pctls_leave_na <- read_sas("pctls_ltfu_190508.sas7bdat")
#pctls_leave_na <- clean_coeff(pctls_leave_na) %>% 
#                  mutate(sex = replace(sex, sex == 'Males', 'male'),
#                         sex = replace(sex, sex == 'Females', 'female')) %>%
#                  unite(group, c('group', 'sex'), remove=TRUE) %>% 
#                  arrange(group)
#
#save(on_art, naaccord, naaccord_prop_2009, init_sqrtcd4n_coeffs,
#     new_dx, new_dx_interval, gmix_param_coeffs, file=paste0(procwd,"/r.convert")) 



