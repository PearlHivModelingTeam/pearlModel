suppressMessages(library(geepack))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(broom))
suppressMessages(library(feather))

input_dir <- filePath(getwd(), '/../../data/input/aim1')
param_dir <- filePath(getwd(), '/../../data/parameters/aim1')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

#####################################################################################
# Model 2: Mortality among those not in care
#####################################################################################
pop1 <- read_sas(filePath(input_dir, 'pop_mortality.sas7bdat'))
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
                    corstr = "exchangeable", 
                    family=binomial(link='logit'))
}

get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- c("intercept", "year", "age_cat", "tv_sqrtcd4n")
  return(coeffs)
}
 
model2 <- yearly %>%
  mutate(model = map(data, model2fx),
         coeffs = map(model, get_coeff),
         vcov = map(model, ~.$geese$vbeta)) %>%
  select(-data)

model2$sex[model2$sex==1] <- "male"
model2$sex[model2$sex==2] <- "female"

model2 <- model2 %>% unite(group, pop2, sex, remove=TRUE)


coeffs <- model2 %>% 
  select(group, coeffs) %>% 
  unnest(cols=coeffs)

vcov <- model2 %>% 
  select(group, vcov) %>% 
  mutate_if(is.list, map, as_data_frame) %>%
  unnest(cols=vcov)

write_feather(coeffs, filePath(param_dir, 'mortality_out_care.feather'))
write_feather(vcov, filePath(param_dir, 'mortality_out_care_vcov.feather'))

