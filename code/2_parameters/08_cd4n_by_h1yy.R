# Load packages ---------------------------------------------------------------
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(feather))
suppressMessages(library(lubridate))
suppressMessages(library(mixtools))

input_dir <- filePath(getwd(), '/../../data/input/aim1')
param_dir <- filePath(getwd(), '/../../data/parameters/aim1')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

test <- read_feather(filePath(input_dir, 'naaccord.feather'))

# Nest by group 
test <- test %>%
  group_by(group) %>%
  nest(.key = "naaccord")

get_cd4n_by_h1yy <- function(naaccord, group) {
  # Go back to NA-ACCORD population & get sqrt cd4n @ HAART initiation
  popu2 <- naaccord %>%
    mutate(H1YY = year(haart1date),
           sqrtcd4n = ifelse(cd4n >= 0, sqrt(cd4n), NA)) %>%
    filter(2010 <= H1YY, H1YY <= 2017)
  
  # Collapse H1YYs into periods for groups with small numbers (IDU only)
  popu2 <- popu2 %>%
     mutate(period = case_when(group %in% c("het_hisp_female", "het_hisp_male") & H1YY %in% c(2015, 2016, 2017) ~ 2015,
                               group %in% c("het_white_female", "het_white_male") & H1YY %in% c(2016, 2017) ~ 2016,
                               group %in% c("idu_black_female") & H1YY %in% c(2012, 2013) ~ 2012,
                               group %in% c("idu_black_female") & H1YY %in% c(2014,2015,2016,2017) ~ 2014,
                               group %in% c("idu_black_male") & H1YY %in% c(2016,2017) ~ 2016,
                               group %in% c("idu_hisp_male") & H1YY %in% c(2012, 2013) ~ 2012,
                               group %in% c("idu_hisp_male") & H1YY %in% c(2015, 2016, 2017) ~ 2015,
                               group %in% c("idu_hisp_female") ~ 2009,
                               group %in% c("idu_white_female") & H1YY %in% c(2009,2010,2011) ~ 2009,
                               group %in% c("idu_white_female") & H1YY %in% c(2012,2013) ~ 2012,
                               group %in% c("idu_white_female") & H1YY %in% c(2014,2015,2016,2017) ~ 2014,
                               group %in% c("idu_white_male") & H1YY %in% c(2016,2017) ~ 2016,
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

cd4n_by_h1yy <- test %>% 
  mutate(cd4n_by_h1yy = pmap(list(naaccord, group), get_cd4n_by_h1yy)) %>%
  select(c(group, cd4n_by_h1yy)) %>% 
  unnest() %>%
  rename_all(tolower)

write_feather(cd4n_by_h1yy, filePath(param_dir, 'cd4n_by_h1yy.feather'))
