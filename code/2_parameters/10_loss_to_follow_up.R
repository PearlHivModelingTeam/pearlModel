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

get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- c('intercept', 'age', '_age', '__age', '___age', 'year', 'sqrtcd4n', 'haart_period')
  return(coeffs)
}

model3 <- yearly %>%
  mutate(model = map(data, model3fx),
         coeffs = map(model, get_coeff),
         vcov = map(model, ~.$geese$vbeta)) %>%
  select(-data)

model3$sex[model3$sex==1] <- "male"
model3$sex[model3$sex==2] <- "female"

model3 <- model3 %>% unite(group, pop2, sex, remove=TRUE)


coeffs <- model3 %>% 
  select(group, coeffs) %>% 
  unnest(cols=coeffs)

vcov <- model3 %>% 
  select(group, vcov) %>% 
  mutate_if(is.list, map, as_data_frame) %>%
  unnest(cols=vcov)

write_feather(coeffs, filePath(param_dir, 'loss_to_follow_up.feather'))
write_feather(vcov, filePath(param_dir, 'loss_to_follow_up_vcov.feather'))

