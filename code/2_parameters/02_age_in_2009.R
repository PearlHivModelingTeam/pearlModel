# Load packages ---------------------------------------------------------------
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(feather))
suppressMessages(library(lubridate))
suppressMessages(library(mixtools))

input_dir <- filePath(getwd(), '/../../data/input/aim_1')
param_dir <- filePath(getwd(), '/../../data/parameters/aim_1')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

test <- read_feather(filePath(input_dir, 'naaccord_2009.feather'))

# Nest by group 
test <- test %>%
  group_by(group) %>%
  nest()

mixed_normal <- function(DF) {
  
  f2 <- normalmixEM2comp(DF$age2009, mu = c(30,50), sigsqrd = c(1,1), lambda = c(0.5,0.5), maxit = 4000)
  
  mus <- c(f2$mu)
  sigmas <- c(f2$sigma)
  lambdas <- c(f2$lambda)
  
  params <- as_tibble(data.frame(mu1 = mus[1],
                       mu2 = mus[2],
                       sigma1 = sigmas[1],
                       sigma2 = sigmas[2],
                       lambda1 = lambdas[1])) %>%
    gather(key='term', value='estimate')


  boot <- boot.se(f2, B=1000, maxit=4000)
  mus <- c(boot$mu.se)
  sigmas <- c(boot$sigma.se)
  lambdas <- c(boot$lambda.se)
  
  boot <- as_tibble(data.frame(mu1 = mus[1],
                       mu2 = mus[2],
                       sigma1 = sigmas[1],
                       sigma2 = sigmas[2],
                       lambda1 = lambdas[1])) %>%
    gather(key='term', value='se')

  params <- params %>%
    add_column(se=boot$se) %>% 
    mutate(conf_high=(estimate + 1.96 * se),
           conf_low=(estimate - 1.96 * se)) %>%
    select(-se)

  return(params)

}

age_in_2009 <- test %>% 
  mutate(age_in_2009 = map(data, mixed_normal))  %>%
  select(c(group, age_in_2009)) %>%
  unnest() %>% 
  rename_all(tolower)

write_feather(age_in_2009, filePath(param_dir, 'age_in_2009.feather'))
