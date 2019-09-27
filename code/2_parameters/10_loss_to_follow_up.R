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
  unnest()

write_feather(model3, filePath(param_dir, 'loss_to_follow_up.feather'))
