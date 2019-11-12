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

get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- c("intercept", "time_from_h1yy", "_time_from_h1yy", "__time_from_h1yy", "___time_from_h1yy", 
                        "cd4cat349", "cd4cat499", "cd4cat500", "agecat",
                        "_timecd4cat349", "_timecd4cat499", "_timecd4cat500",
                        "__timecd4cat349", "__timecd4cat499", "__timecd4cat500",
                        "___timecd4cat349", "___timecd4cat499", "___timecd4cat500")

  return(coeffs)
}


model5 <- yearly %>%
  mutate(model = map(data, model5fx),
         coeffs = map(model, get_coeff),
         vcov = map(model, ~.$geese$vbeta)) %>%
  select(-c(data))

model5$sex[model5$sex==1] <- "male"
model5$sex[model5$sex==2] <- "female"

model5 <- model5 %>% unite(group, pop2, sex, remove=TRUE)

coeffs <- model5 %>% 
  select(group, coeffs) %>% 
  unnest(cols=coeffs)

vcov <- model5 %>% 
  select(group, vcov) %>% 
  mutate_if(is.list, map, as_data_frame) %>%
  unnest(cols=vcov)

write_feather(coeffs, filePath(param_dir, 'cd4_increase.feather'))
write_feather(vcov, filePath(param_dir, 'cd4_increase_vcov.feather'))
