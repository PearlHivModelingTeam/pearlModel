rm(list = ls())
cat("\014")

#' 05/03/19 Re-do rainbow plots to incorporate changes discussed on call with Keri and Parastu 

library(tidyverse)
library(haven)
library(lubridate)
library(forcats)
library(scales)
library(RColorBrewer)
library(openxlsx)
library(R.utils)
library(feather)

datawd1 <- filePath(getwd(), '/../../data/input/')
output_dir <- filePath(getwd(), '/../../out/rainbow_plots')

######################################################################################
# Proportion of NA-ACCORD participants alive at the end of each year
######################################################################################
# Import data
setwd(datawd1)
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

# filter to MSM, IDU, and HET
popu <- popu %>%
  filter(grepl("msm|idu|het", group))

######################################################################################
#' Proportion of NA-ACCORD participants alive at the end of each year: in care only
#' ALL COHORTS
######################################################################################
carestatus <- read_sas("popu16_carestatus.sas7bdat")
colnames(carestatus) <- tolower(colnames(carestatus))

# Aggregate
alive_fx <- function(YEAR) {
  alive <- popu %>%
    mutate(age = YEAR-yob,
           startYY = year(obs_entry),
           stopYY = year(obs_exit),
           H1YY = ifelse(year(haart1date) < 2000, 2000, year(haart1date))) %>%
    filter(startYY <= YEAR, YEAR <= stopYY) %>%
    mutate(calyy = YEAR)
  
  alive <- alive %>%
    bind_rows(alive %>% mutate(group = gsub("_.*", "_all", group)))
  
  # 02/13/19: Merge with the carestatus dataset and filter to patient who were IN CARE in the given year
  alive <- alive %>%
    left_join(carestatus, by = c("naid", "calyy" = "year")) %>%
    filter(in_care==1)
  
  # New 09/11/2018 - drop deaths - our population is patients who survived through to 12/31/YEAR X
  dead <- alive %>%
    filter(year(deathdate)==YEAR)
  
  alive <- alive %>%
    anti_join(dead, by="naid")
  
  alive <- alive %>%
    mutate(agecat = floor(age/10)) %>%
    mutate(agecat = replace(agecat, agecat < 2, 2)) %>%
    mutate(agecat = replace(agecat, agecat > 7, 7))
  
  alive <- alive %>%
    group_by(group, sex, calyy, agecat) %>%
    tally %>%
    ungroup %>%
    group_by(group, sex, calyy) %>%
    mutate(N = sum(n),
           median_pct = n/sum(n)*100) %>%
    ungroup
  
  #alive$agecat <- factor(alive$agecat,
  #levels = c("2", "3", "4", "5", "6", "7"),
  #labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"))
  
  #alive$agecat <- factor(alive$agecat, levels=rev(levels(alive$agecat)))
  
  return(alive)
}

pop09 <- alive_fx(2009)
pop10 <- alive_fx(2010)
pop11 <- alive_fx(2011)
pop12 <- alive_fx(2012)
pop13 <- alive_fx(2013)
pop14 <- alive_fx(2014)
pop15 <- alive_fx(2015)

pop09_15care <- bind_rows(pop09, pop10, pop11, pop12, pop13, pop14, pop15)

# Create 95% CIs - 05/03/19
pop09_15care <- pop09_15care %>%
  mutate(se = sqrt((median_pct/100 * (1-median_pct/100)) / N),
         lower_ci = 100*(median_pct/100 - 1.96*se),
         upper_ci = 100*(median_pct/100 + 1.96*se))

write_feather(pop09_15care, filePath(output_dir, '/naaccord_in_care.feather'))
