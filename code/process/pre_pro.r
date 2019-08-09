######################################################################################
# Load packages
######################################################################################
library(MASS)
library(haven)
library(lubridate)
library(broom)
library(geepack)
library(mixtools)
library(RGeode)
library(triangle)
library(gamlss)
library(binom)
library(tidyverse)

cwd = getwd()
paramwd <- paste0(cwd, '/../../data/param')
naaccordwd <- paste0(cwd, '/../../data/input')
procwd <- paste0(cwd, '/../../data/processed')
outwd <- paste0(cwd, '/../../out')

######################################################################################
# Define group(s) of interest - if multiple, separate by commas
######################################################################################
filtergroup <- c("idu_black", "idu_hisp", "idu_white",
                 "het_black", "het_hisp", "het_white",
                 "msm_black", "msm_hisp", "msm_white")

today <- format(Sys.Date(), format="%y%m%d")

######################################################################################
# Call function source file
######################################################################################
source(paste0(cwd, "/fx.r"))

######################################################################################
# Read in model coefficients from SAS for time-varying processes
######################################################################################
setwd(paramwd)

# CD4 increase
coeff_cd4_increase <- read_sas("cd4_increase_coeff_190508.sas7bdat")
coeff_cd4_increase <- clean_coeff(coeff_cd4_increase)

# CD4 decrease
coeff_cd4_decrease <- read_sas("coeff_cd4_decrease_190508.sas7bdat")
colnames(coeff_cd4_decrease) <- tolower(colnames(coeff_cd4_decrease))

# Mortality for persons out of care
coeff_mortality_out <- read_sas("coeff_mortality_out_care_190508.sas7bdat")
coeff_mortality_out <- clean_coeff(coeff_mortality_out)

# Probability of leaving NA-ACCORD
coeff_leave_na <- read_sas("coeff_ltfu_190508.sas7bdat")
coeff_leave_na <- clean_coeff(coeff_leave_na)

# Percentiles used for age for the spline in the LTFU model
pctls_leave_na <- read_sas("pctls_ltfu_190508.sas7bdat")
pctls_leave_na <- clean_coeff(pctls_leave_na)

######################################################################################
# Read in NA-ACCORD population file and CDC Surveillance Estimates of PLWH in 2009
######################################################################################
# Popu (NA-ACCORD population created in popu.sas)
setwd(naaccordwd)
popu <- read_sas("popu16.sas7bdat")
colnames(popu) <- tolower(colnames(popu))

# Format SAS dates as R dates and filter to males and females
popu <- popu %>%
  mutate(obs_entry = as.Date(obs_entry, origin="1960-01-01"),
         obs_exit = as.Date(obs_exit, origin="1960-01-01"),
         sex = replace(sex, sex==1, "Males"),
         sex = replace(sex, sex==2, "Females"),
         cd4n = replace(cd4n, cd4n < 0, NA)) %>%
  filter(sex %in% c("Males", "Females"), pop2!="") %>%
  rename(group = pop2)

popu$group <- tolower(popu$group)

# CHNAGE 09/13/18 - RECODE STUDY EXIT TO BE BASED ON MIN(DEATHDATE, COHORT_CLOSEDATE)
popu <- popu %>%
  mutate(obs_exit2 = pmin(deathdate, cohort_closedate, na.rm = T)) %>%
  select(-obs_exit) %>%
  rename(obs_exit = obs_exit2)

# CHANGE 10/30/18 - SET DEATHDATE TO MISSING IF REASON FOR EXIT IS NOT DEATH
popu <- popu %>%
  mutate(deathdate = replace(deathdate, deathdate!=obs_exit, NA))

# Nest by group and sex
popu <- popu %>%
  group_by(group, sex) %>%
  nest %>%
  rename(data_popu = data)

# Persons living with diagnosed HIV infection on HAART in 2009
art2009 <- read.csv("surveillance_estimates_cdc_2009.csv", stringsAsFactors = FALSE)
colnames(art2009) <- tolower(colnames(art2009))

# Nest by group and sex
art2009 <- art2009 %>%
  mutate(on_art = floor(n_alive_2009*pct_art*.01)) %>%
  select(group, sex, on_art) %>%
  group_by(group, sex) %>%
  nest %>%
  rename(data_art2009 = data)

# Merge popu dataset with art2009 dataset
test <- popu %>%
  left_join(art2009, by=c("group", "sex"))

######################################################################################
#' Filter datasets to population of interest - filter by DF name and group
#' filterfx <- function(DF, VALUE)
######################################################################################
test <- filterfx(test, filtergroup)

# Check that correct groups were selected
test %>% select(group) %>% distinct

######################################################################################
# Get Mortality param est - specify name of df WITHIN the nested df to run
######################################################################################
# Map the mortality function to the NA-ACCORD data frame to get coefficient estimates (only will be used for patients in care)
test <- test %>%
  mutate(mortality = map(data_popu, mortality_fx))

# Unnest to add the coefficients as separate columns to the dataset
test <- test %>%
  unnest(mortality)

######################################################################################
#' 2009 population of persons in the US living with diagnosed HIV and on HAART -
#'      TIME-FIXED COMPONENTS
######################################################################################
# 0. Get number of persons alive on HAART (obtained from CDC HIV Surveillance reports and MMWR reports)
test <- test  %>%
  mutate(on_art = map_dbl(data_art2009, dplyr::pull))

# 1. Get population of NA-ACCORD particiapnts alive in 2009
test <- test %>%
  mutate(naaccord_2009 = map(data_popu, fx1, naaccordwd))

# 2. Fit a weibull distribution to age in 2009
test <- test %>%
  mutate(weibull_2009 = map(naaccord_2009, fx2))

# 3. Fit a mixed normal distribution to age in 2009
test <- test %>%
  mutate(mixture_2009 = map(naaccord_2009, fx3))

# 05/02/19: IDU Hisp female hack - don't allow mixed normal
test0 <- test %>% filter(group=="idu_hisp", sex=="Females") %>% select(group, sex, mixture_2009) %>% unnest 
test0$lambda1 <- 0
test0 <- test0 %>%
  group_by(group, sex) %>%
  nest(.key = "mixture_2009")

test0 <- test %>% 
  filter(group=="idu_hisp", sex=="Females") %>% 
  select(-mixture_2009) %>%
  left_join(test0, by = c("group", "sex"))

test <- test %>% 
  anti_join(test0, by = c("group", "sex")) %>%
  bind_rows(test0)

# 4. Get proportion by age in NA-ACCORD population
test <- test %>%
  mutate(naaccord_prop_2009 = map(naaccord_2009, fx4))

# 5. Get Mean/SD of CD4N at HAART initiation - stratified by H1YY - updated 04/24/19
test <- test %>%
  #mutate(naaccord_cd4_2009 = map(data_popu, fx5))
  mutate(naaccord_cd4_2009 = map(naaccord_2009, fx5))

######################################################################################
# 2009 - 2030 population of new HIV Dx's - TIME-FIXED COMPONENTS
######################################################################################
# Read in dataset of # of new HIV diagnoses, 2009-2015 reported by CDC in Table 1
surv <- read.csv("dx_estimates_cdc_table1.csv", stringsAsFactors = FALSE)
colnames(surv) <- tolower(colnames(surv))

surv <- surv_fx1(surv)

surv2 <- surv_fx2(surv)

test <- test %>%
  left_join(surv %>%
              group_by(group, sex) %>%
              nest(.key="surv"), by = c("group", "sex")) %>%
  left_join(surv2 %>%
              group_by(group, sex) %>%
              nest(.key="hiv_pred_interval"), by = c("group", "sex"))

######################################################################################
# 2009 - 2015 population of new INIs - TIME-FIXED COMPONENTS
######################################################################################
# 1. Fit a mixed normal distribution to age at HAART initiation for each year, 2009-2015 - point of diff from MSM model
test <- test %>%
  mutate(ini1 = pmap(list(data_popu, group, sex), inifx1))

# 2. Fit a GLM to each parameter from the MIXTURE model and extract predicted values of mu, sigma, and lambda parameters
test <- test %>%
  mutate(ini2 = map(ini1, inifx2))

save.image(file=paste0(procwd, '/processed.rda'))
