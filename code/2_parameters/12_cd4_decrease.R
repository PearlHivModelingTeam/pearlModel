suppressMessages(library(geepack))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(broom))
suppressMessages(library(feather))

input_dir <- filePath(getwd(), '/../../data/input/aim_1')
param_dir <- filePath(getwd(), '/../../data/parameters/aim_1')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

#####################################################################################
# Model 4: CD4 decline
#####################################################################################
# Read in analysis population

get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- c('intercept', 'time_out_of_naaccord', 'sqrtcd4n_exit')
  return(coeffs)
}

pop1 <- read_sas(filePath(input_dir, 'pop_cd4_decrease.sas7bdat'))
colnames(pop1) <- tolower(colnames(pop1))

model <- glm(diff ~ time_out_of_naaccord + sqrtcd4_exit,
              data = pop1)

coeffs = as_tibble(get_coeff(model))
vcov = as_tibble(vcov(model))

write_feather(coeffs, filePath(param_dir, 'cd4_decrease.feather'))
write_feather(vcov, filePath(param_dir, 'cd4_decrease_vcov.feather'))
