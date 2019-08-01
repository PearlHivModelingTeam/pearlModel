rm(list = ls())
cat("\014")

library(tidyverse)
library(haven)
library(lubridate)
library(forcats)
library(scales)
library(RColorBrewer)
library(openxlsx)

#######################################################################################
# Compare the static version to the sensitivity analysis version
#######################################################################################
outwd <- "L:\\naaccord\\Silver\\Data\\simulation\\All\\190625\\plots"
datawd1 <- "L:\\naaccord\\Silver\\Data\\naaccord" # NA-ACCORD pop files

# Static 
hetwd <- "L:\\naaccord\\Silver\\Data\\simulation\\All\\190517\\HET"
iduwd <- "L:\\naaccord\\Silver\\Data\\simulation\\All\\190517\\IDU"
msmwd <- "L:\\naaccord\\Silver\\Data\\simulation\\All\\190517\\MSM"

# Sensitivity analysis
hetwd2 <- "L:\\naaccord\\Silver\\Data\\simulation\\All\\190625\\HET"
iduwd2 <- "L:\\naaccord\\Silver\\Data\\simulation\\All\\190625\\IDU"
msmwd2 <- "L:\\naaccord\\Silver\\Data\\simulation\\All\\190625\\MSM"

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
# Load simulated data and re-shape - want in_care and newly_lost
######################################################################################
simfx <- function(wd, racesex, dfname) {
  setwd(paste0(wd, "\\", racesex))
  
  y <- get(load(dfname))
  
  y <- unlist(y, recursive = F)
  
  df1 <- y[names(y)=="newly_lost"]
  names(df1) <- seq(1,length(df1))
  df1 <- bind_rows(df1, .id = "dataset") %>% ungroup
  
  df2 <- y[names(y)=="in_care"]
  names(df2) <- seq(1,length(df2))
  df2 <- bind_rows(df2, .id = "dataset") %>% ungroup
  
  df <- bind_rows(df1, df2)
  
  df <- df %>%
    group_by(group, sex, dataset, calyy, agecat) %>%
    summarise(total_n = sum(n)) %>%
    ungroup %>%
    group_by(group, sex, dataset, calyy) %>%
    mutate(pct = total_n/sum(total_n)*100) %>%
    ungroup
  
}

# Get simulated pop(static)
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

# Get simulated pop (sensitivity analysis)
all_sensitivity <- bind_rows(# HET
  simfx(hetwd2, "bf", "test.76.Rda"),
  simfx(hetwd2, "bm", "test.90.Rda"),
  simfx(hetwd2, "hf", "test.77.Rda"),
  simfx(hetwd2, "hm", "test.85.Rda"),
  simfx(hetwd2, "wf", "test.73.Rda"),
  simfx(hetwd2, "wm", "test.99.Rda"),
  # IDU
  simfx(iduwd2, "bf", "test.57.Rda"),
  simfx(iduwd2, "bm", "test.32.Rda"),
  simfx(iduwd2, "hf", "test.49.Rda"),
  simfx(iduwd2, "hm", "test.89.Rda"),
  simfx(iduwd2, "wf", "test.72.Rda"),
  simfx(iduwd2, "wm", "test.38.Rda"),
  # MSM
  simfx(msmwd2, "bm", "test.72.Rda")#,
  #simfx(msmwd2, "hm", "test.84.Rda"),
  #simfx(msmwd2, "wm", "test.59.Rda")
)

all <- all %>%
  mutate(version = "Base") %>%
  bind_rows(all_sensitivity %>% mutate(version = "Sensitivity analysis"))

######################################################################################
# Get the median and 95% CIs in the simulated data - IN CARE ONLY
######################################################################################
# Bind simulated and observed data
overall <- pop09_15care %>%
  mutate(source = 1) %>%
  bind_rows(all %>% mutate(source = 2))

# Format factors
overall$source <- factor(overall$source,
                          levels = c(1,2),
                          labels = c("NA-ACCORD (All Cohorts)", 
                                     "Simulated"))

overall$agecat <- factor(overall$agecat,
                         levels = c("2", "3", "4", "5", "6", "7"),
                         labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"))

overall$agecat <- factor(overall$agecat, levels=rev(levels(overall$agecat)))

chk <- overall %>%
  filter(group=="idu_black", sex=="Males", source=="Simulated")

chk2 <- overall %>%
  filter(group=="idu_black", sex=="Males", source=="NA-ACCORD (All Cohorts)")

chk2 <- chk2 %>% mutate(version = "Base") %>% bind_rows(chk2 %>% mutate(version = "Sensitivity analysis"))

p <- ggplot(data = chk, aes(x = calyy, y = pct, group = dataset, color = dataset)) +
  facet_grid(version~agecat) +
  geom_line() +
  scale_color_viridis_d() +
  theme(legend.position = "none")

p + geom_line(data = chk2, aes(x = calyy, y = median_pct), color = "red", inherit.aes = F) 


# Nest by group and sex
overall <- overall %>%
  group_by(group, sex) %>%
  nest

# Define the graphing function
ggfx <- function(DF, group, sex) {
  chk <- DF %>%
    filter(source=="Simulated")
  
  chk2 <- DF %>%
    filter(source=="NA-ACCORD (All Cohorts)")
  
  chk2 <- chk2 %>% mutate(version = "Base") %>% bind_rows(chk2 %>% mutate(version = "Sensitivity analysis"))
  
  p <- ggplot(data = chk, aes(x = calyy, y = pct, group = dataset, color = dataset)) +
    facet_grid(version~agecat) +
    geom_line(alpha = 0.5) +
    scale_color_viridis_d() +
    theme_bw() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste0(group, " - ", sex))
  
  p <- p + geom_line(data = chk2, aes(x = calyy, y = median_pct), lwd = 1, color = "red", inherit.aes = F) 
  
}

overall2 <- overall %>%
  mutate(plot = pmap(list(data, group, sex), ggfx),
         filename = paste0(group, "_", sex, "_in_care_line_plot", ".png")) %>%
  select(filename, plot)

setwd(outwd)
pwalk(overall2, ggsave, path =  getwd(), height = 8, width = 13)

######################################################################################
# GRAPH VERSION 2: IQRs displayeds
######################################################################################
overall <- overall %>%
  unnest

# Get the median and IQR values in the simulated data
iqr <- overall %>%
  filter(source=="Simulated") %>%
  group_by(source, group, sex, calyy, agecat, version) %>%
  summarise(median_pct = median(pct),
            p25_pct = quantile(pct, probs = 0.25),
            p75_pct = quantile(pct, probs = 0.75)) %>%
  ungroup

iqr <- iqr %>%
  bind_rows(overall %>% filter(source=="NA-ACCORD (All Cohorts)"))

chk <- iqr %>%
  filter(group=="idu_black", sex=="Males", source=="Simulated")

chk2 <- iqr %>%
  filter(group=="idu_black", sex=="Males", source=="NA-ACCORD (All Cohorts)")

chk2 <- chk2 %>% mutate(version = "Base") %>% bind_rows(chk2 %>% mutate(version = "Sensitivity analysis"))

p <- ggplot(data = chk, aes(x = calyy, y = median_pct)) +
  facet_grid(version~agecat) +
  geom_line() +
  geom_ribbon(aes(ymin = p25_pct, ymax = p75_pct), alpha = 0.3, fill = "yellow") +
  theme(legend.position = "none")

p + geom_line(data = chk2, aes(x = calyy, y = median_pct), color = "red", inherit.aes = F) 

# Nest by group and sex
iqr <- iqr %>%
  group_by(group, sex) %>%
  nest

# Define the graphing function
ggfx <- function(DF, group, sex) {
  chk <- DF %>%
    filter(source=="Simulated")
  
  chk2 <- DF %>%
    filter(source=="NA-ACCORD (All Cohorts)")
  
  chk2 <- chk2 %>% mutate(version = "Base") %>% bind_rows(chk2 %>% mutate(version = "Sensitivity analysis"))
  
  p <- ggplot(data = chk, aes(x = calyy, y = median_pct)) +
    facet_grid(version~agecat) +
    geom_line() +
    geom_ribbon(aes(ymin = p25_pct, ymax = p75_pct), alpha = 0.3, fill = "yellow") +
    theme_bw() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste0(group, " - ", sex))
  
  p <- p + geom_line(data = chk2, aes(x = calyy, y = median_pct), color = "red", inherit.aes = F) 
  
}

iqr2 <- iqr %>%
  mutate(plot = pmap(list(data, group, sex), ggfx),
         filename = paste0(group, "_", sex, "_in_care_line_plot_iqr", ".png")) %>%
  select(filename, plot)

setwd(outwd)
pwalk(iqr2, ggsave, path =  getwd(), height = 8, width = 13)

######################################################################################
# Graph raw #'s for each population
######################################################################################
simfx2 <- function(wd, racesex, dfname) {
  setwd(paste0(wd, "\\", racesex))
  
  y <- get(load(dfname))
  
  y <- unlist(y, recursive = F)
  
  datasets <- c("dead_out_care", "dead_care", "newly_lost", "newly_reengage", "in_care", "out_care")
  alldatasets <- vector("list", 6)
  names(alldatasets) <- datasets
  
  for (i in seq_along(datasets)) {
    df1 <- y[names(y)==datasets[i]]
    names(df1) <- seq(1,length(df1))
    df1 <- bind_rows(df1, .id = "dataset") %>% 
      ungroup %>%
      select(dataset, group, sex, calyy, agecat, n)
    
    alldatasets[[i]] <- df1
  }
  
  alldatasets <- bind_rows(alldatasets, .id = "population")
  
  return(alldatasets)
  
}

# Get simulated pop(static)
all <- bind_rows(# HET
  simfx2(hetwd, "bf", "test.76.Rda"),
  simfx2(hetwd, "bm", "test.90.Rda"),
  simfx2(hetwd, "hf", "test.77.Rda"),
  simfx2(hetwd, "hm", "test.85.Rda"),
  simfx2(hetwd, "wf", "test.73.Rda"),
  simfx2(hetwd, "wm", "test.99.Rda"),
  # IDU
  simfx2(iduwd, "bf", "test.57.Rda"),
  simfx2(iduwd, "bm", "test.32.Rda"),
  simfx2(iduwd, "hf", "test.49.Rda"),
  simfx2(iduwd, "hm", "test.89.Rda"),
  simfx2(iduwd, "wf", "test.72.Rda"),
  simfx2(iduwd, "wm", "test.38.Rda"),
  # MSM
  simfx2(msmwd, "bm", "test.72.Rda"),
  simfx2(msmwd, "hm", "test.84.Rda"),
  simfx2(msmwd, "wm", "test.59.Rda"))

# Get simulated pop (sensitivity analysis)
all_sensitivity <- bind_rows(# HET
  simfx2(hetwd2, "bf", "test.76.Rda"),
  simfx2(hetwd2, "bm", "test.90.Rda"),
  simfx2(hetwd2, "hf", "test.77.Rda"),
  simfx2(hetwd2, "hm", "test.85.Rda"),
  simfx2(hetwd2, "wf", "test.73.Rda"),
  simfx2(hetwd2, "wm", "test.99.Rda"),
  # IDU
  simfx2(iduwd2, "bf", "test.57.Rda"),
  simfx2(iduwd2, "bm", "test.32.Rda"),
  simfx2(iduwd2, "hf", "test.49.Rda"),
  simfx2(iduwd2, "hm", "test.89.Rda"),
  simfx2(iduwd2, "wf", "test.72.Rda"),
  simfx2(iduwd2, "wm", "test.38.Rda"),
  # MSM
  simfx2(msmwd2, "bm", "test.72.Rda")#,
  #simfx2(msmwd2, "hm", "test.84.Rda"),
  #simfx2(msmwd2, "wm", "test.59.Rda")
)

all <- all %>%
  mutate(version = "Base") %>%
  bind_rows(all_sensitivity %>% mutate(version = "Sensitivity analysis"))

all$agecat <- factor(all$agecat,
                         levels = c("2", "3", "4", "5", "6", "7"),
                         labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"))

all$agecat <- factor(all$agecat, levels=rev(levels(all$agecat)))


# Create combo in_care and newly_lost
add1 <- all %>%
  filter(population %in% c("in_care", "newly_lost")) %>%
  group_by(dataset, group, sex, calyy, agecat, version) %>%
  summarise(total_n = sum(n)) %>%
  ungroup %>%
  mutate(population = "on_art") %>%
  select(population, dataset, group, sex, calyy, agecat, n = total_n, version)

# Create combo out_care and newly_reengage
add2 <- all %>%
  filter(population %in% c("out_care", "newly_reengage")) %>%
  group_by(dataset, group, sex, calyy, agecat, version) %>%
  summarise(total_n = sum(n)) %>%
  ungroup %>%
  mutate(population = "off_art") %>%
  select(population, dataset, group, sex, calyy, agecat, n = total_n, version)

# Combine and drop in_care, out_care
all2 <- all %>%
  filter(!population %in% c("in_care", "out_care")) %>%
  bind_rows(add1, add2)

# Get median and IQR of n
all3 <- all2 %>%
  group_by(version, population, group, sex, calyy, agecat) %>%
  summarise(median_n = median(n),
            p25_n = quantile(n, probs = 0.25),
            p75_n = quantile(n, probs = 0.75)) %>%
  ungroup

# Graph
all3_g <- all3 %>%
  group_by(group, sex, population) %>%
  nest

ggfx <- function(DF, group, sex, population) {
  p <- ggplot(data = DF, aes(x = calyy, y = median_n)) +
    facet_grid(version ~ agecat) +
    geom_line() +
    geom_ribbon(aes(ymin = p25_n, ymax = p75_n), alpha = 0.3, fill = "yellow") +
    theme_bw() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = "Year", 
         y = "# of simulated people",
         title = paste0(group, " - ", sex),
         subtitle = population)
}

all3_g <- all3_g %>%
  mutate(plot = pmap(list(data, group, sex, population), ggfx),
         filename = paste0(group, "_", sex, "_", population, "_total_by_age_iqr", ".png")) %>%
  select(filename, plot)

setwd(outwd)
pwalk(all3_g, ggsave, path =  getwd(), height = 8, width = 13)
write.csv(all3, file = "data_total_by_age_iqr.csv", row.names = F)

# Re-graph showing the numbers overall across all age groups
all3 <- all2 %>%
  group_by(dataset, version, population, group, sex, calyy) %>%
  summarise(total_n = sum(n)) %>%
  ungroup %>%
  group_by(version, population, group, sex, calyy) %>%
  summarise(median_n = median(total_n),
            p25_n = quantile(total_n, probs = 0.25),
            p75_n = quantile(total_n, probs = 0.75)) %>%
  ungroup

# Graph
all3_g <- all3 %>%
  group_by(group, sex, population) %>%
  nest

ggfx <- function(DF, group, sex, population) {
  p <- ggplot(data = DF, aes(x = calyy, y = median_n)) +
    facet_grid(version ~ .) +
    geom_line() +
    geom_ribbon(aes(ymin = p25_n, ymax = p75_n), alpha = 0.3, fill = "yellow") +
    theme_bw() +
    ylim(0, NA) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = "Year", 
         y = "# of simulated people",
         title = paste0(group, " - ", sex),
         subtitle = population)
}

all3_g <- all3_g %>%
  mutate(plot = pmap(list(data, group, sex, population), ggfx),
         filename = paste0(group, "_", sex, "_", population, "_total_iqr", ".png")) %>%
  select(filename, plot)

setwd(outwd)
pwalk(all3_g, ggsave, path =  getwd(), height = 8, width = 13)
write.csv(all3, file = "data_total_iqr.csv", row.names = F)


######################################################################################
# Number of new HAART initiators
######################################################################################
simfx3 <- function(wd, racesex, dfname) {
  setwd(paste0(wd, "\\", racesex))
  
  y <- get(load(dfname))
  
  y <- unlist(y, recursive = F)
  
  df1 <- y[names(y)=="newart_age"]
  names(df1) <- seq(1,length(df1))
  df1 <- bind_rows(df1, .id = "dataset") %>% ungroup
  
  df1 <- df1 %>%
    mutate(agecat = floor(age/10), 
           agecat = replace(agecat, agecat < 2, 2),
           agecat = replace(agecat, agecat > 7, 7)) %>%
    group_by(dataset, group, sex, agecat, h1yy) %>%
    summarise(total_n = sum(n)) %>%
    ungroup %>%
    rename(n = total_n)
  
}

# Get simulated pop(static)
all <- bind_rows(# HET
  simfx3(hetwd, "bf", "test.76.Rda"),
  simfx3(hetwd, "bm", "test.90.Rda"),
  simfx3(hetwd, "hf", "test.77.Rda"),
  simfx3(hetwd, "hm", "test.85.Rda"),
  simfx3(hetwd, "wf", "test.73.Rda"),
  simfx3(hetwd, "wm", "test.99.Rda"),
  # IDU
  simfx3(iduwd, "bf", "test.57.Rda"),
  simfx3(iduwd, "bm", "test.32.Rda"),
  simfx3(iduwd, "hf", "test.49.Rda"),
  simfx3(iduwd, "hm", "test.89.Rda"),
  simfx3(iduwd, "wf", "test.72.Rda"),
  simfx3(iduwd, "wm", "test.38.Rda"),
  # MSM
  simfx3(msmwd, "bm", "test.72.Rda"),
  simfx3(msmwd, "hm", "test.84.Rda"),
  simfx3(msmwd, "wm", "test.59.Rda"))

# Get simulated pop (sensitivity analysis)
all_sensitivity <- bind_rows(# HET
  simfx3(hetwd2, "bf", "test.76.Rda"),
  simfx3(hetwd2, "bm", "test.90.Rda"),
  simfx3(hetwd2, "hf", "test.77.Rda"),
  simfx3(hetwd2, "hm", "test.85.Rda"),
  simfx3(hetwd2, "wf", "test.73.Rda"),
  simfx3(hetwd2, "wm", "test.99.Rda"),
  # IDU
  simfx3(iduwd2, "bf", "test.57.Rda"),
  simfx3(iduwd2, "bm", "test.32.Rda"),
  simfx3(iduwd2, "hf", "test.49.Rda"),
  simfx3(iduwd2, "hm", "test.89.Rda"),
  simfx3(iduwd2, "wf", "test.72.Rda"),
  simfx3(iduwd2, "wm", "test.38.Rda"),
  # MSM
  simfx3(msmwd2, "bm", "test.72.Rda")#,
  #simfx3(msmwd2, "hm", "test.84.Rda"),
  #simfx3(msmwd2, "wm", "test.59.Rda")
)

all <- all %>%
  mutate(version = "Base") %>%
  bind_rows(all_sensitivity %>% mutate(version = "Sensitivity analysis"))

all$agecat <- factor(all$agecat,
                     levels = c("2", "3", "4", "5", "6", "7"),
                     labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"))

all$agecat <- factor(all$agecat, levels=rev(levels(all$agecat)))

# Get median and IQR of n
all2 <- all %>%
  group_by(version, group, sex, h1yy, agecat) %>%
  summarise(median_n = median(n),
            p25_n = quantile(n, probs = 0.25),
            p75_n = quantile(n, probs = 0.75)) %>%
  ungroup

# Graph - by age
all2_g <- all2 %>%
  group_by(group, sex) %>%
  nest

ggfx <- function(DF, group, sex) {
  p <- ggplot(data = DF, aes(x = h1yy, y = median_n)) +
    facet_grid(version ~ agecat) +
    geom_line() +
    geom_ribbon(aes(ymin = p25_n, ymax = p75_n), alpha = 0.3, fill = "yellow") +
    theme_bw() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = "Year", 
         y = "# of simulated people",
         title = paste0(group, " - ", sex))
}

all2_g <- all2_g %>%
  mutate(plot = pmap(list(data, group, sex), ggfx),
         filename = paste0(group, "_", sex, "_", "_haart_initiators_by_age_iqr", ".png")) %>%
  select(filename, plot)

setwd(outwd)
pwalk(all2_g, ggsave, path =  getwd(), height = 8, width = 13)
write.csv(all2, file = "data_haart_initiators_by_age_iqr.csv", row.names = F)

# Graph - overall
all2 <- all %>%
  group_by(dataset, version, group, sex, h1yy) %>%
  summarise(total_n = sum(n)) %>%
  ungroup %>%
  group_by(version, group, sex, h1yy) %>%
  summarise(median_n = median(total_n),
            p25_n = quantile(total_n, probs = 0.25),
            p75_n = quantile(total_n, probs = 0.75)) %>%
  ungroup

all2_g <- all2 %>%
  group_by(group, sex) %>%
  nest

ggfx <- function(DF, group, sex) {
  p <- ggplot(data = DF, aes(x = h1yy, y = median_n)) +
    facet_grid(version ~ .) +
    geom_line() +
    geom_ribbon(aes(ymin = p25_n, ymax = p75_n), alpha = 0.3, fill = "yellow") +
    theme_bw() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = "Year", 
         y = "# of simulated people",
         title = paste0(group, " - ", sex)) +
    ylim(0, NA)
}

all2_g <- all2_g %>%
  mutate(plot = pmap(list(data, group, sex), ggfx),
         filename = paste0(group, "_", sex, "_", "_haart_initiators_total_iqr", ".png")) %>%
  select(filename, plot)

setwd(outwd)
pwalk(all2_g, ggsave, path =  getwd(), height = 8, width = 13)
write.csv(all2, file = "data_haart_initiators_total_iqr.csv", row.names = F)
