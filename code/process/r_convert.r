######################################################################################
# Load packages
######################################################################################

#suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(haven))
suppressPackageStartupMessages(library(lubridate))
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
# Call function source file and define functions
######################################################################################
source(paste0(wd,"/fx.r"))
options(dplyr.print_max = 100)

ini_cd4_fx2 <- function(NAACCORD, GROUP, SEX) {
  # Go back to NA-ACCORD population & get sqrt cd4n @ HAART initiation
  popu2 <- NAACCORD %>%
    mutate(H1YY = year(haart1date),
           sqrtcd4n = ifelse(cd4n >= 0, sqrt(cd4n), NA)) %>%
    filter(2010 <= H1YY, H1YY <= 2014)
  
  # Collapse H1YYs into periods for groups with small numbers (IDU only)
  popu2 <- popu2 %>%
    mutate(period = case_when(GROUP %in% c("idu_black", "idu_white") & SEX=="Females" & H1YY %in% c(2009, 2010) ~ 2009,
                              GROUP %in% c("idu_black", "idu_white")  & SEX=="Females" & H1YY %in% c(2011, 2012) ~ 2011,
                              GROUP %in% c("idu_black", "idu_white")  & SEX=="Females" & H1YY %in% c(2013, 2014, 2015) ~ 2013,
                              GROUP %in% c("idu_hisp") & SEX=="Males" & H1YY %in% c(2009, 2010) ~ 2009,
                              GROUP %in% c("idu_hisp")  & SEX=="Males" & H1YY %in% c(2011, 2012) ~ 2011,
                              GROUP %in% c("idu_hisp")  & SEX=="Males" & H1YY %in% c(2013, 2014, 2015) ~ 2013,
                              GROUP=="idu_hisp" & SEX=="Females" ~ 2009,
                              TRUE ~ H1YY))
  
  # Get the mean sqrt CD4N by H1YY (MODIFIED FOR IDU TO USE PERIOD INSTEAD OF H1YY)
  sumdat <- popu2 %>%
    filter(!is.na(sqrtcd4n)) %>%
    group_by(period) %>%
    summarise(mean = mean(sqrtcd4n),
              sd = sd(sqrtcd4n)) %>%
    ungroup
  
  # Fit GLMs to the mean and SD (MODIFIED FOR IDU TO USE PERIOD INSTEAD OF H1YY)
  meandat <- glm(sumdat$mean ~ sumdat$period)
  stddat <- glm(sumdat$sd ~ sumdat$period)
  
  meanint <- meandat$coefficients[1]
  meanslp <- meandat$coefficients[2]
  stdint <- stddat$coefficients[1]
  stdslp <- stddat$coefficients[2]
  
  # Update 03/27/19: set slopes to 0 for IDU Hisp F
  # For Hispanic IDU Females - coalesce w/ 0

  params <- data.frame(meanint = meandat$coefficients[1],
                       meanslp = meandat$coefficients[2],
                       stdint = stddat$coefficients[1],
                       stdslp = stddat$coefficients[2])

  params[is.na(params)] <- 0

  return(params)
}

######################################################################################
# Main Function
######################################################################################

# Load Functions
indir <- paste0(wd,"/../../data/processed")
load(paste0(procwd,"/processed.rda"))

test <- test %>% 
  mutate(init_sqrtcd4n_coeff = pmap(list(data_popu, group, sex), ini_cd4_fx2))

test <- test %>% mutate(sex = replace(sex, sex == 'Males', 'male'),
                        sex = replace(sex, sex == 'Females', 'female')) %>%
                 unite(group, c('group', 'sex'), remove=TRUE) %>% 
                 arrange(group)


# on_art
on_art <- test %>% select(group, on_art)

# mixture_2009
mixture_2009_coeff <- test %>% select(group, mixture_2009) %>% unnest

# naaccord
naaccord <- test %>% select(group, data_popu) %>% unnest

# naaccord_prop_2009
naaccord_prop_2009 <- test %>% select(group, naaccord_prop_2009) %>% unnest

# init_sqrtcd4n_coeff_2009
init_sqrtcd4n_coeff_2009 <- test %>% select(group, naaccord_cd4_2009) %>% unnest

# coeff_age_2009_ci
#coeff_age_2009_ci <- test %>% select(group, age2009_ci) %>% unnest

# new_dx
new_dx <- test %>% select(group, surv) %>% unnest

# new_dx_interval
new_dx_interval <- test %>% select(group, hiv_pred_interval) %>% unnest

# mixture_h1yy_coeff
mixture_h1yy_coeff <- test%>% select(group, ini2) %>% unnest # %>% spread(param, pred)

# init_sqrtcd4n_coeff
init_sqrtcd4n_coeff <- test %>% select(group, init_sqrtcd4n_coeff) %>% unnest

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
#save(on_art, naaccord, naaccord_prop_2009, init_sqrtcd4n_coeff_2009,
#     new_dx, new_dx_interval, gmix_param_coeffs, file=paste0(procwd,"/r.convert")) 



