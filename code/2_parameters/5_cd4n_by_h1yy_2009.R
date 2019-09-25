# Load packages ---------------------------------------------------------------
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(feather))
suppressMessages(library(lubridate))

input_dir <- filePath(getwd(), '/../../data/input')
param_dir <- filePath(getwd(), '/../../data/parameters')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

test <- read_feather(filePath(input_dir, 'naaccord_2009.feather'))

# Nest by group 
test <- test %>%
  group_by(group) %>%
  nest(.key = "naaccord_2009")

# Get Mean/SD of CD4N at H1YY - stratified by H1YY ----------------------------
get_cd4n_by_h1yy <- function(NAACCORD) {
  popu2 <- NAACCORD %>%
    mutate(startYY = year(obs_entry),
           stopYY = year(obs_exit),
           H1YY = year(haart1date),
           sqrtcd4n = ifelse(cd4n >= 0, sqrt(cd4n), NA)) %>%
    filter(startYY <= 2009, 2009 <= stopYY)
  
  # Get mean and SD sqrtcd4n by H1YY - changed from 2013 to 2009 8/24
  outdat <- popu2 %>%
    filter(H1YY >= 2000, H1YY <= 2009, sqrtcd4n >= 0) %>%
    group_by(H1YY) %>% 
    summarise(sqrtcd4n_mean = mean(sqrtcd4n),
              sqrtcd4n_sd = sd(sqrtcd4n),
              sqrtcd4n_n = n()) %>%
    ungroup
  
  # FIT GLM TO MEAN AND SD OF SQRTCD4N
  meandat <- glm(outdat$sqrtcd4n_mean ~ outdat$H1YY)
  stddat <- glm(outdat$sqrtcd4n_sd ~ outdat$H1YY)
  
  params <- data.frame(meanint = meandat$coefficients[1],
                       meanslp = meandat$coefficients[2],
                       stdint = stddat$coefficients[1],
                       stdslp = stddat$coefficients[2])
}


cd4n_by_h1yy_2009 <- test %>% 
  mutate(cd4n_by_h1yy_2009 = map(naaccord_2009, get_cd4n_by_h1yy)) %>%
  select(c(group, cd4n_by_h1yy_2009)) %>%
  unnest() %>% 
  rename_all(tolower)

write_feather(cd4n_by_h1yy_2009, filePath(param_dir, 'cd4n_by_h1yy_2009.feather'))
