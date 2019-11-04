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
# Model 2: Mortality among those not in care
#####################################################################################
pop1 <- read_sas(filePath(input_dir, 'pop_mortality_190508.sas7bdat'))
colnames(pop1) <- tolower(colnames(pop1))
pop1$pop2 <- tolower(pop1$pop2)

yearly <- pop1 %>%
  select(pop2, sex, naid, realdeath, year, agecat, tv_sqrtcd4n) %>%  
  na.omit() %>%
  group_by(pop2, sex) %>%
  nest

model2fx <- function(DF) {
  mylogit <- geeglm(realdeath ~ year + agecat + tv_sqrtcd4n, 
                    id = naid, 
                    data = DF, 
                    corstr = "unstructured", 
                    family=binomial(link='logit'))
}

tidy_estimates <- function(model) {
  estimates <- tidy(model, conf.int = TRUE, conf.level = 0.005) %>%
    mutate(term = c('intercept', 'year', 'age_cat', 'tv_sqrtcd4n')) %>%
    #select(c(term, estimate, conf.low, conf.high)) %>%
    rename(conf_low = conf.low, conf_high = conf.high)
  print(estimates)
  estimates <- estimates %>% select(c(term, estimate, conf_low, conf_high))
}

model2 <- yearly %>%
  mutate(model = map(data, model2fx),
         estimates = map(model, tidy_estimates)) %>%
  select(-c(data, model))

model2$sex[model2$sex==1] <- "male"
model2$sex[model2$sex==2] <- "female"

model2 <- model2 %>% 
  unite(group, pop2, sex) %>%
  unnest()

write_feather(model2, filePath(param_dir, 'mortality_out_care.feather'))

