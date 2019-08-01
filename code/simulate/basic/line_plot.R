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

outwd <- "/home/cameron/SilverTsunami/out"
datawd1 <- "/home/cameron/SilverTsunami/liz/naaccord/Silver/Data/naaccord" # NA-ACCORD pop files
hetwd <- paste0(outwd,"/HET")
iduwd <- paste0(outwd,"/IDU") 
msmwd <- paste0(outwd,"/MSM")

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


######################################################################################
#' Proportion of NA-ACCORD participants alive at the end of each year: in care only
#' HIVRN DROPPED
######################################################################################
# Aggregate
alive_fx <- function(YEAR) {
  alive <- popu %>%
    filter(cohort!=30) %>%
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

pop09_15care_hivrn <- bind_rows(pop09, pop10, pop11, pop12, pop13, pop14, pop15)

# Create 95% CIs - 05/03/19
pop09_15care_hivrn <- pop09_15care_hivrn %>%
  mutate(se = sqrt((median_pct/100 * (1-median_pct/100)) / N),
         lower_ci = 100*(median_pct/100 - 1.96*se),
         upper_ci = 100*(median_pct/100 + 1.96*se))

######################################################################################
# Load simulated data and re-shape - want in_care and newly_lost
######################################################################################
simfx <- function(wd, racesex, dfname) {
  setwd(paste0(wd, "/", racesex))
  
  y <- get(load(dfname))
  
  y <- unlist(y, recursive = F)
  
  df1 <- y[names(y)=="newly_lost"]
  names(df1) <- seq(1,10)
  df1 <- bind_rows(df1, .id = "dataset") %>% ungroup
  
  df2 <- y[names(y)=="in_care"]
  names(df2) <- seq(1,10)
  df2 <- bind_rows(df2, .id = "dataset") %>% ungroup
  
  df <- bind_rows(df1, df2)
  
  df <- df %>%
    group_by(dataset, group, sex, calyy, agecat) %>%
    summarise(total = sum(n)) %>%
    ungroup
  
  overall <- df %>%
    group_by(group, sex, calyy, agecat) %>%
    summarise(median_n = floor(median(total)),
              p025_n = floor(quantile(total, probs = 0.025)),
              p975_n = floor(quantile(total, probs = 0.975))) %>%
    ungroup %>%
    group_by(group, sex, calyy) %>%
    mutate(median_N = sum(median_n),
           p025_N = sum(p025_n),
           p975_N = sum(p975_n),
           median_pct = median_n/median_N*100,
           lower_ci = p025_n/p025_N*100,
           upper_ci = p975_n/p975_N*100) %>%
    ungroup 
}

# Get simulated pop
all <- bind_rows(# HET
                 simfx(hetwd, "bf", "test.76.Rda"),
                 simfx(hetwd, "bm", "test.90.Rda"),
                 simfx(hetwd, "hf", "test.77.Rda"),
                 simfx(hetwd, "hm", "test.85.Rda"),
                 simfx(hetwd, "wf", "test.73.Rda"),
                 simfx(hetwd, "wm", "test.99.Rda"),
                 # IDU
                 simfx(iduwd, "bf", "test.57.Rda"),
                 simfx(iduwd, "bm", "test.32.Rda"),
                 simfx(iduwd, "hf", "test.49.Rda"),
                 simfx(iduwd, "hm", "test.89.Rda"),
                 simfx(iduwd, "wf", "test.72.Rda"),
                 simfx(iduwd, "wm", "test.38.Rda"),
                 # MSM
                 simfx(msmwd, "bm", "test.72.Rda"),
                 simfx(msmwd, "hm", "test.84.Rda"),
                 simfx(msmwd, "wm", "test.59.Rda"))

######################################################################################
# Get the median and 95% CIs in the simulated data - IN CARE ONLY
######################################################################################
# Re-calculate numbers for "All" groups
all2 <- all %>% 
  mutate(group = gsub("_.*", "_all", group))

all2 <- all2 %>%
  group_by(group, sex, calyy, agecat) %>%
  summarise(median_n = sum(median_n),
            p025_n = sum(p025_n),
            p975_n = sum(p975_n),
            median_N = sum(median_N),
            p025_N = sum(p025_N),
            p975_N = sum(p975_N),
            median_pct = median_n/median_N*100,
            lower_ci = p025_n / p025_N*100,
            upper_ci = p975_n/p975_N*100) %>%
  ungroup
  
all <- bind_rows(all, all2)

# Bind simulated and observed data
overall <- pop09_15care %>%
  mutate(dataset = 1) %>%
  bind_rows(pop09_15care_hivrn %>% mutate(dataset = 2)) %>%
  bind_rows(all %>% mutate(dataset = 3))

# Format factors
overall$dataset <- factor(overall$dataset,
                          levels = c(1,2,3),
                          labels = c("NA-ACCORD (All Cohorts)", 
                                     "NA-ACCORD (without HIVRN)",
                                     "Simulated"))

overall$agecat <- factor(overall$agecat,
                         levels = c("2", "3", "4", "5", "6", "7"),
                         labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"))

overall$agecat <- factor(overall$agecat, levels=rev(levels(overall$agecat)))


# Nest by group and sex
overall <- overall %>%
  group_by(group, sex) %>%
  nest

# Define the graphing function
ggfx <- function(DF, group, sex) {
  ggplot(data = DF, aes(x = calyy, y = median_pct)) +
    facet_wrap(~agecat) +
    geom_line(data = DF %>% filter(dataset=="Simulated")) +
    geom_ribbon(data = DF %>% filter(dataset=="Simulated"), aes(ymin = lower_ci, ymax = upper_ci), alpha = 0.5) +
    geom_point(data = DF %>% filter(dataset=="NA-ACCORD (All Cohorts)"), aes(color = dataset)) +
    geom_point(data = DF %>% filter(dataset=="NA-ACCORD (without HIVRN)"), aes(color = dataset)) +
    geom_linerange(data = DF %>% filter(dataset=="NA-ACCORD (All Cohorts)"),
                   aes(ymin = lower_ci, ymax = upper_ci, color = dataset)) +
    geom_linerange(data = DF %>% filter(dataset=="NA-ACCORD (without HIVRN)"),
                   aes(ymin = lower_ci, ymax = upper_ci, color = dataset)) +
    theme_classic() +
    theme(legend.position = "bottom") +
    labs(title = "Dots = NA-ACCORD, Lines = Simulated",
         subtitle = paste0(group, " - ", sex),
         x = "Year",
         y = "%") 
  
}

overall2 <- overall %>%
  mutate(plot = pmap(list(data, group, sex), ggfx),
         filename = paste0(group, "_", sex, "_in_care_line_plot", ".png")) %>%
  select(filename, plot)

setwd(outwd)
pwalk(overall2, ggsave, path =  getwd(), height = 8, width = 13)

# Raw dataset
overallx <- overall %>%
  unnest %>%
  filter(calyy <= 2015) #%>%
  #select(-se) %>%
  #rename(pct = median_pct)

#write.csv(overallx, file = "in_care_line_plot_data.csv", row.names = F)
