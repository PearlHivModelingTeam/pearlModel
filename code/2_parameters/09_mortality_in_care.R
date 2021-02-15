suppressMessages(library(geepack))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(broom))

pearl_dir <- Sys.getenv("PEARL_DIR")
input_dir <- filePath(pearl_dir, '/param/raw')
intermediate_dir <- filePath(pearl_dir, '/param/intermediate')
param_dir <- filePath(pearl_dir, '/param/param')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

#####################################################################################
# Model 1: Mortality among those in care
#####################################################################################
# Read in analysis population

pop1 <- read_sas(filePath(input_dir, 'pop_mortality.sas7bdat'))
colnames(pop1) <- tolower(colnames(pop1))
pop1$pop2 <- tolower(pop1$pop2)

yearly <- pop1 %>%
  mutate(ageby10 = floor(age/10),
         ageby10 = replace(ageby10, ageby10==1, 2),
         ageby10 = replace(ageby10, ageby10 > 7, 7)) %>%
  filter(year >= 2009, # modified 9/26/18
         year <= 2017) %>% # added 09/11/18
  select(pop2, sex, naid, realdeath, year, ageby10, sqrtcd4n, h1yy) %>%  
  na.omit() %>%
  group_by(pop2, sex) %>% 
  nest()


modelfx <- function(df) {
  mylogit <- geeglm(realdeath ~ year + ageby10 + sqrtcd4n + h1yy, 
                    id = naid, 
                    data = df, 
                    corstr = "exchangeable",
                    family=binomial(link='logit'))
}



get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- c("intercept", "year", "age_cat", "sqrtcd4n", "h1yy")
  return(coeffs)
}

model2 <- yearly %>%
  mutate(model = map(data, modelfx),
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
  mutate_if(is.list, map, as_tibble, .name_repair='unique') %>%
  unnest(cols=vcov)

write_csv(coeffs, filePath(param_dir, 'mortality_in_care.csv'))
write_csv(vcov, filePath(param_dir, 'mortality_in_care_vcov.csv'))
