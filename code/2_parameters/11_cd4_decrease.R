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

write_feather(model4, filePath(param_dir, 'cd4_decrease.feather'))

