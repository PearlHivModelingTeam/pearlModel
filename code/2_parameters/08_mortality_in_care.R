suppressMessages(library(geepack))
suppressMessages(library(lubridate))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(broom))
suppressMessages(library(feather))

input_dir <- filePath(getwd(), '/../../data/input')
param_dir <- filePath(getwd(), '/../../data/parameters')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')


mortality_fx <- function(DF) {
  # Create new variables
  patient <- DF %>%
    mutate(h1yy = year(haart1date),
           sqrtcd4n = ifelse(cd4n >=0, sqrt(cd4n), NA),
           deathyy = year(deathdate),
           entryyy = year(obs_entry),
           exityy = year(obs_exit))
  
  # Convert from wide to long - 1 row / year of study observation for each patient
  yearly <- patient %>%
    nest(entryyy, exityy) %>%
    mutate(year = map(data, ~seq(unique(.x$entryyy), unique(.x$exityy), 1))) %>%
    unnest(year)
  
  # Recode variables
  yearly <- yearly %>%
    mutate(age = year - yob,
           agecat = floor(age/10), 
           agecat = replace(agecat, agecat < 2, 2),
           agecat = replace(agecat, agecat > 6, 6),
           ageby10 = floor(age/10),
           ageby10 = replace(ageby10, ageby10==1, 2),
           ageby10 = replace(ageby10, ageby10 > 7, 7),
           realdeath = 0,
           realdeath = replace(realdeath, year==deathyy, 1),
           py = ifelse(year == year(obs_entry) & year == year(obs_exit), (obs_exit - obs_entry + 1) / 365.25,
                       ifelse(year == year(obs_entry), (make_date(year = year, month = 12, day = 31) - obs_entry + 1) / 365.25,
                              ifelse(year==year(obs_exit), (obs_exit - make_date(year = year, mont = 1, day = 1) + 1) / 365.25, 1))),
           logpy = log(py)) %>%
    filter(year >= 2009, # modified 9/26/18
           year <= 2015) # added 09/11/18
  
  # Drop patients with any missing predictor variable that will be used in regression
  yearly2 <- yearly %>%
    select(naid, realdeath, ageby10, year, h1yy, sqrtcd4n) %>%  
    na.omit() 

  # Run regression model
  modelfx <- function(df) {
    mylogit <- geeglm(realdeath ~ ageby10 + sqrtcd4n + year + h1yy, 
                      id = naid, 
                      data = df, 
                      corstr = "unstructured", 
                      family=binomial(link='logit'))
  }

  tidy_estimates <- function(model) {
    estimates <- tidy(model, conf.int = TRUE) %>%
      mutate(term = c('intercept_est', 'ageby10_est', 'sqrtcd4n_est', 'year_est', 'h1yy_est')) %>%
      select(c(term, estimate, conf.low, conf.high)) %>%
      rename(conf_low = conf.low, conf_high = conf.high)
  }
 
  model = modelfx(yearly2)
  estimates = tidy_estimates(model)

}

test <- read_feather(filePath(input_dir, 'naaccord.feather')) %>% group_by(group) %>% nest()

test <- test %>%
  mutate(mortality = map(data, mortality_fx))

test <- test %>% unnest(mortality) %>% select(-data)

write_feather(test, filePath(param_dir, 'mortality_in_care.feather'))
