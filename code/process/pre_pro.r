######################################################################################
# Load packages
######################################################################################

suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(haven))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(broom))
suppressPackageStartupMessages(library(geepack))
suppressPackageStartupMessages(library(mixtools))
suppressPackageStartupMessages(library(RGeode))
suppressPackageStartupMessages(library(triangle))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(binom))
suppressPackageStartupMessages(library(tidyverse))

wd <- getwd()

paramwd <- paste0(wd, "/../../data/param")
naaccordwd <- paste0(wd, "/../../data/input")
outwd <- paste0(wd, "/../../data/processed")

######################################################################################
# Call function source file
######################################################################################
source(paste0(wd,"/fx.r"))

setwd(paramwd)
# 95% CIs for age params in 2009
age2009_mixture_ci <- get(load("age2009_mixture_ci.rda"))

######################################################################################
# Define group(s) of interest - if multiple, separate by commas
######################################################################################
filtergroup <- c("idu_black", "idu_hisp", "idu_white",
                 "het_black", "het_hisp", "het_white",
                 "msm_black", "msm_hisp", "msm_white")
#filtergroup <- c("het_black")

today <- format(Sys.Date(), format="%y%m%d")


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
setwd(paramwd)
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
print("Groups to be simulated:")
test %>% select(group) %>% distinct

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
#test <- test %>%
#  mutate(weibull_2009 = map(naaccord_2009, fx2))

# 3. Fit a mixed normal distribution to age in 2009
#test <- test %>%
#  mutate(mixture_2009 = map(naaccord_2009, fx3))

# 05/02/19: IDU Hisp female hack - don't allow mixed normal
#test0 <- test %>% filter(group=="idu_hisp", sex=="Females") %>% select(group, sex, mixture_2009) %>% unnest 
#test0$lambda1 <- 0
#test0 <- test0 %>%
#  group_by(group, sex) %>%
#  nest(.key = "mixture_2009")

#test0 <- test %>% 
#  filter(group=="idu_hisp", sex=="Females") %>% 
#  select(-mixture_2009) %>%
#  left_join(test0, by = c("group", "sex"))

#test <- test %>% 
#  anti_join(test0, by = c("group", "sex")) %>%
#  bind_rows(test0)
# 4. Get proportion by age in NA-ACCORD population
test <- test %>%
  mutate(naaccord_prop_2009 = map(naaccord_2009, fx4))

# 5. Get Mean/SD of CD4N at HAART initiation - stratified by H1YY - updated 04/24/19
test <- test %>%
  #mutate(naaccord_cd4_2009 = map(data_popu, fx5))
  mutate(naaccord_cd4_2009 = map(naaccord_2009, fx5))

# 6. 95% CIs of mixed normal parameters for age in 2009
test <- test %>%
  left_join(age2009_mixture_ci %>%
              group_by(group, sex) %>%
              nest(.key = "age2009_ci"), by = c("group", "sex"))

######################################################################################
# 2009 - 2030 population of new HIV Dx's - TIME-FIXED COMPONENTS
######################################################################################
# Read in dataset of # of new HIV diagnoses, 2009-2015 reported by CDC in Table 1
setwd(paramwd)
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
# 1. Fit a mixed normal distribution to age at HAART initiation for each year, 2009-2015
test <- test %>%
  mutate(ini1 = pmap(list(data_popu, group, sex), inifx1))

# 2. Fit a GLM to each parameter from the MIXTURE model and extract predicted values of mu, sigma, and lambda parameters
test <- test %>%
  mutate(ini2 = map(ini1, inifx2))

######################################################################################
# Save preprocessed data                                                             #
######################################################################################

save(test, paramwd, file=paste0(outwd,"/sense.data")) 
