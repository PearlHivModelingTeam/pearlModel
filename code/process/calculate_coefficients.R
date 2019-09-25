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

#####################################################################################
# Model 2: Mortality among those not in care
#####################################################################################
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
  estimates <- tidy(model, conf.int = TRUE) %>%
    mutate(term = c('intercept', 'year', 'age_cat', 'tv_sqrtcd4n')) %>%
    select(c(term, estimate, conf.low, conf.high)) %>%
    rename(conf_low = conf.low, conf_high = conf.high)
}

model2 <- yearly %>%
  mutate(model = map(data, model2fx),
         estimates = map(model, tidy_estimates)) %>%
  select(-c(data, model))

model2$sex[model2$sex==1] <- "male"
model2$sex[model2$sex==2] <- "female"

model2 <- model2 %>% 
  unite(group, pop2, sex) %>%
  unnest() %>% 
  mutate(func = 'mortality_out_care')

write_feather(model2, filePath(param_dir, 'mortality_out_care.feather'))

######################################################################################
## Model 3: Prob of LTF
######################################################################################
# Read in analysis population

pop1 <- read_sas(filePath(input_dir, 'pop_ltfu_190508.sas7bdat'))
colnames(pop1) <- tolower(colnames(pop1))
pop1$pop2 <- tolower(pop1$pop2)

# Get vars of interest
yearly <- pop1 %>%
  select(pop2, sex, naid, out_of_care, age, `_age`, `__age`, `___age`, year, sqrtcd4n, haart_period) %>%  
  na.omit() %>%
  group_by(pop2, sex) %>%
  nest

# Model function
model3fx <- function(DF) {
  mylogit <- geeglm(out_of_care ~ age + `_age` + `__age` + `___age` + year + sqrtcd4n + haart_period, 
                    id = naid, 
                    data = DF, 
                    corstr = "exchangeable", 
                    family=binomial(link='logit'))
}

tidy_estimates <- function(model) {
  estimates <- tidy(model, conf.int = TRUE) %>%
    mutate(term = c('intercept', 'age', '_age', '__age', '___age', 'year', 'sqrt_cd4n', 'haart_period')) %>%
    select(c(term, estimate, conf.low, conf.high)) %>%
    rename(conf_low = conf.low, conf_high = conf.high)
}

model3 <- yearly %>%
  mutate(model = map(data, model3fx),
         estimates = map(model, tidy_estimates)) %>%
  select(-c(data, model))

model3$sex[model3$sex==1] <- "male"
model3$sex[model3$sex==2] <- "female"

model3 <- model3 %>% 
  unite(group, pop2, sex) %>%
  unnest() %>% 
  mutate(func = 'loss_to_follow_up')

write_feather(model3, filePath(param_dir, 'loss_to_follow_up.feather'))

#####################################################################################
# Model 4: CD4 decline
#####################################################################################
# Read in analysis population

pop1 <- read_sas(filePath(input_dir, 'pop_cd4_decrease_190508.sas7bdat'))
colnames(pop1) <- tolower(colnames(pop1))

model <- glm(diff ~ time_out_of_naaccord + sqrtcd4_exit,
              data = pop1)

estimates <- tidy(model, conf.int = TRUE) %>%
  mutate(term = c('intercept', 'time_out_of_naaccord', 'sqrtcd4n_exit')) %>%
  select(c(term, estimate, conf.low, conf.high)) %>%
  rename(conf_low = conf.low, conf_high = conf.high)

model4 <- tbl_df(data.frame())
for (group_name in group_names) {
  model4 <- bind_rows(model4, mutate(estimates, group = group_name))
}

model4 <-mutate(model4, func = 'cd4_decrease')

write_feather(model4, filePath(param_dir, 'cd4_decrease.feather'))

######################################################################################
## Model 5: CD4 increase
######################################################################################
pop1 <- read_sas(filePath(input_dir, 'pop_cd4_increase_190508.sas7bdat'))
colnames(pop1) <- tolower(colnames(pop1))
pop1$pop2 <- tolower(pop1$pop2)

# Get vars of interest
yearly <- pop1 %>%
  select(pop2, sex, naid, sqrtcd4n, ends_with("time_from_h1yy"), starts_with("cd4cat"), agecat) %>%  
  na.omit() %>%
  group_by(pop2, sex) %>%
  nest

# Model function
model5fx <- function(DF) {
  mylogit <- geeglm(sqrtcd4n ~ time_from_h1yy + `_time_from_h1yy` + `__time_from_h1yy` + `___time_from_h1yy` + 
                      cd4cat349 + cd4cat499 + cd4cat500 + 
                      agecat +
                      `_time_from_h1yy`:cd4cat349 +
                      `_time_from_h1yy`:cd4cat499 +
                      `_time_from_h1yy`:cd4cat500 +
                      `__time_from_h1yy`:cd4cat349 +
                      `__time_from_h1yy`:cd4cat499 +
                      `__time_from_h1yy`:cd4cat500 +
                      `___time_from_h1yy`:cd4cat349 +
                      `___time_from_h1yy`:cd4cat499 +
                      `___time_from_h1yy`:cd4cat500, 
                    id = naid, 
                    data = DF, 
                    corstr = "exchangeable", 
                    family=gaussian(link='identity'))
}

tidy_estimates <- function(model) {
  terms <- c("intercept", "time_from_h1yy", "_time_from_h1yy", "__time_from_h1yy", "___time_from_h1yy", 
                               "cd4cat349", "cd4cat499", "cd4cat500", "agecat",
                               "_timecd4cat349", "_timecd4cat499", "_timecd4cat500",
                               "__timecd4cat349", "__timecd4cat499", "__timecd4cat500",
                               "___timecd4cat349", "___timecd4cat499", "___timecd4cat500")
  estimates <- tidy(model, conf.int = TRUE) %>%
    mutate(term = terms) %>%
    select(c(term, estimate, conf.low, conf.high)) %>%
    rename(conf_low = conf.low, conf_high = conf.high)
}

model5 <- yearly %>%
  mutate(model = map(data, model5fx),
         estimates = map(model, tidy_estimates)) %>%
  select(-c(data, model))

model5$sex[model5$sex==1] <- "male"
model5$sex[model5$sex==2] <- "female"

model5 <- model5 %>% 
  unite(group, pop2, sex) %>%
  unnest() %>% 
  mutate(func = 'cd4_increase')

write_feather(model5, filePath(param_dir, 'cd4_increase.feather'))

