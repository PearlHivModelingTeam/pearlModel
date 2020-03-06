# Load packages ---------------------------------------------------------------
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))

input_dir <- filePath(getwd(), '/../../data/input/aim_1')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

# Functions -------------------------------------------------------------------
collapse_groups <- function(df) {
  df <- df %>%
    mutate(sex = replace(sex, sex==1, 'male'),
           sex = replace(sex, sex==2, 'female'),
           pop2 = tolower(pop2)) %>%
    filter(sex %in% c('male', 'female'), pop2!='') %>%
    unite(group, pop2, sex)
}

# Read in and format NAACCORD data --------------------------------------------
naaccord <- read_sas(filePath(input_dir, 'popu17.sas7bdat')) %>%
  rename_all(tolower) %>%
  collapse_groups() %>%
  mutate(obs_entry = as.Date(obs_entry, origin="1960-01-01"),
         obs_exit = as.Date(obs_exit, origin="1960-01-01"),
         cd4n = replace(cd4n, cd4n < 0, NA)) %>%
  select(-c(risk, race, pop1))

# Set deathdate to missing if reason for exit is not death
naaccord <- mutate(naaccord, deathdate = replace(deathdate, deathdate!=obs_exit, NA))

write_feather(naaccord, filePath(input_dir, 'naaccord.feather'))
