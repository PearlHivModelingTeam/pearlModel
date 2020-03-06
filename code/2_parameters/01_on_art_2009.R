# Load packages ---------------------------------------------------------------
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))

input_dir <- filePath(getwd(), '/../../data/input/aim_1')
param_dir <- filePath(getwd(), '/../../data/parameters/aim_1')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')


# Get count of people on ART from CDC data ----------------------------------------------- 
on_art <- read.csv(filePath(input_dir, 'surveillance_estimates_cdc_2009.csv'), stringsAsFactors = FALSE) %>%
  rename_all(tolower) %>%
  mutate(sex = replace(sex, sex=='Males', 'male'),
         sex = replace(sex, sex=='Females', 'female')) %>%
  unite(group, group, sex) %>% 
  mutate(on_art = floor(n_alive_2009*pct_art*.01)) %>%
  select(group, on_art)

write_feather(on_art, filePath(param_dir, 'on_art_2009.feather'))
