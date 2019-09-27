suppressMessages(library(geepack))
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

#####################################################################################
# Model 1: Mortality among those in care
#####################################################################################
# Read in analysis population

pop1 <- read_sas(filePath(input_dir, 'pop_mortality_190508.sas7bdat'))
colnames(pop1) <- tolower(colnames(pop1))
pop1$pop2 <- tolower(pop1$pop2)

yearly <- pop1 %>%
  select(pop2, sex, naid, realdeath, year, agecat, sqrtcd4n, h1yy) %>%  
  na.omit() %>%
  group_by(pop2, sex) %>%
  nest()


modelfx <- function(df) {
  mylogit <- geeglm(realdeath ~ year + agecat + sqrtcd4n + h1yy, 
                    id = naid, 
                    data = df, 
                    corstr = "unstructured", 
                    family=binomial(link='logit'))
}

tidy_estimates <- function(model) {
  estimates <- tidy(model, conf.int = TRUE) %>%
    mutate(term = c('intercept_est', 'year_est', 'ageby10_est', 'sqrtcd4n_est', 'h1yy_est')) %>%
    select(c(term, estimate, conf.low, conf.high)) %>%
    rename(conf_low = conf.low, conf_high = conf.high)
}

model1 <- yearly %>%
  mutate(model = map(data, modelfx),
         estimates = map(model, tidy_estimates)) %>% 
  select(-c(data, model))

model1$sex[model1$sex==1] <- "male"
model1$sex[model1$sex==2] <- "female"

model1 <- model1 %>% 
  unite(group, pop2, sex) %>%
  unnest()

write_feather(model1, filePath(param_dir, 'mortality_in_care.feather'))
