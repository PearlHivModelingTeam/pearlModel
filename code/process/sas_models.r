library(geepack)
library(haven)
library(tidyverse)

wd <- getwd()
modelwd <- paste0(wd, "/../../data/param")

#####################################################################################
# Model 1: Mortality among those in care
#####################################################################################
# Read in analysis population
setwd(modelwd)

pop1 <- read_sas("pop_mortality_190508.sas7bdat")
colnames(pop1) <- tolower(colnames(pop1))
pop1$pop2 <- tolower(pop1$pop2)

yearly <- pop1 %>%
  select(pop2, sex, naid, realdeath, year, agecat, sqrtcd4n, h1yy) %>%  
  na.omit() %>%
  group_by(pop2, sex) %>%
  nest

modelfx <- function(DF) {
  mylogit <- geeglm(realdeath ~ year + agecat + sqrtcd4n + h1yy, 
                    id = naid, 
                    data = DF, 
                    corstr = "unstructured", 
                    family=binomial(link='logit'))
}

get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- paste0(c("intercept", "year", "agecat", "sqrtcd4n", "h1yy"), "_c")
  return(coeffs)
}

model1 <- yearly %>%
  mutate(model = map(data, modelfx),
         coeffs = map(model, get_coeff),
         vcov = map(model, ~.$geese$vbeta)) %>%
  select(-data) %>%
  rename(group = pop2)

model1$sex[model1$sex==1] <- "Males"
model1$sex[model1$sex==2] <- "Females"

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

get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- paste0(c("intercept", "year", "agecat", "tv_sqrtcd4n"), "_c")
  return(coeffs)
}

model2 <- yearly %>%
  mutate(model = map(data, model2fx),
         coeffs = map(model, get_coeff),
         vcov = map(model, ~.$geese$vbeta)) %>%
  select(-data)  %>%
  rename(group = pop2)

model2$sex[model2$sex==1] <- "Males"
model2$sex[model2$sex==2] <- "Females"

#####################################################################################
# Model 3: Prob of LTF
#####################################################################################
# Read in analysis population
setwd(modelwd)

pop1 <- read_sas("pop_ltfu_190508.sas7bdat")
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
  colnames(coeffs) <- paste0(c("intercept", "age", "_age", "__age", "___age", "year", "sqrtcd4n", "haart_period"), "_c")
  return(coeffs)
}

# Map model function and pull coeffs and vcov
model3 <- yearly %>%
  mutate(model = map(data, model3fx),
         coeffs = map(model, get_coeff),
         vcov = map(model, ~.$geese$vbeta)) %>%
  select(-data) %>%
  rename(group = pop2)

model3$sex[model3$sex==1] <- "Males"
model3$sex[model3$sex==2] <- "Females"

#####################################################################################
# Model 4: CD4 decline
#####################################################################################
# Read in analysis population
setwd(modelwd)

pop1 <- read_sas("pop_cd4_decrease_190508.sas7bdat")
colnames(pop1) <- tolower(colnames(pop1))

model <- glm(diff ~ time_out_of_naaccord + sqrtcd4_exit,
              data = pop1)

coeff <- coef(model)
coeff <- data.frame(t(coeff))
colnames(coeff) <- paste0(c("intercept", "time_out_of_naaccord", "sqrtcd4_exit"), "_c")

vcov <- vcov(model)

model4 <- list(model, coeff, vcov)
names(model4) <- c("model", "coeff", "vcov")

#####################################################################################
# Model 5: CD4 increase
#####################################################################################
# Read in analysis population
setwd(modelwd)

pop1 <- read_sas("pop_cd4_increase_190508.sas7bdat")
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
  colnames(coeffs) <- paste0(c("intercept", "time_from_h1yy", "_time_from_h1yy", "__time_from_h1yy", "___time_from_h1yy", 
                               "cd4cat349", "cd4cat499", "cd4cat500", "agecat",
                               "_timecd4cat349", "_timecd4cat499", "_timecd4cat500",
                               "__timecd4cat349", "__timecd4cat499", "__timecd4cat500",
                               "___timecd4cat349", "___timecd4cat499", "___timecd4cat500"), "_c")
  return(coeffs)
}


# Map model function and pull coeffs and vcov
model5 <- yearly %>%
  mutate(model = map(data, model5fx),
         coeffs = map(model, get_coeff),
         vcov = map(model, ~.$geese$vbeta)) %>%
  select(-data) %>%
  rename(group = pop2)

model5$sex[model5$sex==1] <- "Males"
model5$sex[model5$sex==2] <- "Females"

#####################################################################################
# Save
#####################################################################################
setwd(modelwd)

save(model1, file = "coeff_mortality_in_care_190508.rda")
save(model2, file = "coeff_mortality_out_care_190508.rda")
save(model3, file = "coeff_ltfu_190508.rda")
save(model4, file = "coeff_cd4_decrease_190508.rda")
save(model5, file = "coeff_cd4_increase_190508.rda")


