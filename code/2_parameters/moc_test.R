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
  mutate(ageby10 = floor(age/10),
         ageby10 = replace(ageby10, ageby10==1, 2),
         ageby10 = replace(ageby10, ageby10 > 7, 7)) %>%
  filter(year >= 2009, # modified 9/26/18
         year <= 2015) %>% # added 09/11/18
  select(pop2, sex, naid, realdeath, year, ageby10, sqrtcd4n, h1yy) %>%  
  na.omit() %>%
  group_by(pop2, sex) %>% 
  nest()


modelfx <- function(df) {
  mylogit <- geeglm(realdeath ~ year + ageby10 + sqrtcd4n + h1yy, 
                    id = naid, 
                    data = df, 
                    corstr = "unstructured", 
                    family=binomial(link='logit'))
}

tidy_estimates <- function(model) {
  estimates <- tidy(model, conf.int = TRUE, conf.level=0.005) %>%
    mutate(term = c('intercept_est', 'year_est', 'ageby10_est', 'sqrtcd4n_est', 'h1yy_est')) %>%
    #select(c(term, estimate, conf.low, conf.high)) %>%
    rename(conf_low = conf.low, conf_high = conf.high)
  print(estimates)
  estimates <- estimates %>% select(c(term, estimate, conf_low, conf_high))
}


get_coeff <- function(DF) {
  coeffs <- coef(DF)
  coeffs <- data.frame(t(coeffs))
  colnames(coeffs) <- paste0(c("intercept", "year", "ageby10", "sqrtcd4n", "h1yy"), "_est")
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
  mutate_if(is.list, map, as_data_frame) %>%
  unnest(cols=vcov)

write_feather(coeffs, filePath(param_dir, 'mortality_in_care.feather'))
write_feather(vcov, filePath(param_dir, 'mortality_in_care_vcov.feather'))


#model <- modelfx(yearly)
#estimates <- tidy_estimates(model)
#newdat <- tibble(int = c(1,1), year=c(2010, 2011), ageby10=c(2,7), sqrtcd4n=c(12.1, 16.2), h1yy = c(2007, 2000))
#
#Xp <- model.matrix(delete.response(terms(model)), newdat)
#V <- model$geese$vbeta
#b <- coef(model)
#print(Xp)
#print(b)
#print(V)
#
#yh <- c(Xp %*% b)
#print(yh)
#var.fit <- rowSums((Xp %*% V) * Xp)
#print(sqrt(var.fit))
#print(model$df)
#alpha <- 0.95  ## 90%
#Qt <- c(-1, 1) * qt((1 - alpha) /2, model$df, lower.tail=FALSE)
#print(Qt)
#CI <- yh + outer(sqrt(var.fit), Qt)
#colnames(CI) <- c("lwr", "upr")
#print(Qt)
#print(CI)


#model1 <- yearly %>%
#  mutate(model = map(data, modelfx),
#         estimates = map(model, tidy_estimates)) %>% 
#  select(-c(data, model))
#
#model1$sex[model1$sex==1] <- "male"
#model1$sex[model1$sex==2] <- "female"
#
#model1 <- model1 %>% 
#  unite(group, pop2, sex) %>%
#  unnest()
#
#write_feather(model1, filePath(param_dir, 'mortality_in_care.feather'))
