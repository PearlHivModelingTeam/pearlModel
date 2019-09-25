# Load packages ---------------------------------------------------------------
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(feather))
suppressMessages(library(lubridate))

options(tibble.print_max = 1000)

input_dir <- filePath(getwd(), '/../../data/input')
param_dir <- filePath(getwd(), '/../../data/parameters')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

load(filePath(input_dir, 'age2009_mixture_ci.Rda')) 
age_in_2009 <- age2009_mixture_ci %>% 
  mutate(sex = replace(sex, sex=="Females", 'female'),
         sex = replace(sex, sex=='Males', 'male')) %>%
  unite(group, group, sex)

age_in_2009 <- age_in_2009 %>%
  gather(key = 'term', value = 'est', -group) %>%
  separate(col=term, into=c('term', 'ci'), sep='_') %>% 
  spread(ci, est) %>%
  rename(conf_low = p025, conf_high = p975) %>%
  mutate(estimate = rowMeans(cbind(conf_low, conf_high)))

write_feather(age_in_2009, filePath(param_dir, 'age_in_2009.feather'))
