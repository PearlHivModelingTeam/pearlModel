# Load packages ---------------------------------------------------------------
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(lubridate))
suppressMessages(library(mixtools))

pearl_dir <- Sys.getenv("PEARL_DIR")
input_dir <- filePath(pearl_dir, '/param/raw')
intermediate_dir <- filePath(pearl_dir, '/param/intermediate')
validation_dir <- filePath(pearl_dir, '/param/validation')
param_dir <- filePath(pearl_dir, '/param/param')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

test <- read_csv(filePath(intermediate_dir, 'naaccord.csv'))

# Nest by group 
test <- test %>%
  group_by(group) %>%
  nest(.key = "naaccord")

# Fit a mixed normal distribution to age at HAART initiation for each year, 2009-2015
get_age_by_h1yy <- function(NAACCORD, group) {
  inipop <- NAACCORD %>%
    mutate(H1YY = year(haart1date),
           iniage = H1YY-yob) %>%
    filter(2010 <= H1YY, H1YY <= 2017)

 inipop <- inipop %>%
    mutate(period = case_when(group %in% c("het_hisp_female", "het_hisp_male") & H1YY %in% c(2015, 2016, 2017) ~ 2015,
                              group %in% c("het_white_female", "het_white_male") & H1YY %in% c(2016, 2017) ~ 2016,
                              group %in% c("idu_black_female") & H1YY %in% c(2012, 2013) ~ 2012,
                              group %in% c("idu_black_female") & H1YY %in% c(2014,2015,2016,2017) ~ 2014,
                              group %in% c("idu_black_male") & H1YY %in% c(2016,2017) ~ 2016,
                              group %in% c("idu_hisp_male") & H1YY %in% c(2012, 2013) ~ 2012,
                              group %in% c("idu_hisp_male") & H1YY %in% c(2015, 2016, 2017) ~ 2015,
                              group %in% c("idu_hisp_female") ~ 2010,
                              group %in% c("idu_white_female") & H1YY %in% c(2010,2011) ~ 2010,
                              group %in% c("idu_white_female") & H1YY %in% c(2012,2013) ~ 2012,
                              group %in% c("idu_white_female") & H1YY %in% c(2014,2015,2016,2017) ~ 2014,
                              group %in% c("idu_white_male") & H1YY %in% c(2016,2017) ~ 2016,
                              TRUE ~ H1YY))

  inipop2 <- inipop %>%
    arrange(period) %>%
    group_by(period) %>%
    nest %>%
    mutate(mixture_model = map(data, ~(normalmixEM(maxit = 4000, .$iniage, k = 2, mu = c(25, 50), sigma = c(1,1), lambda = c(0.5, 0.5)))),
           mus = map(mixture_model, ~(.$mu)),
           sigmas = map(mixture_model, ~(.$sigma)),
           lambdas = map(mixture_model, ~(.$lambda)))
  
  inipop2 <- inipop2 %>%
    mutate(mu1 = map_dbl(mus, ~.[1]),
           mu2 = map_dbl(mus, ~.[2]),
           sigma1 = map_dbl(sigmas, ~.[1]),
           sigma2 = map_dbl(sigmas, ~.[2]),
           lambda1 = map_dbl(lambdas, ~.[1]),
           lambda2 = map_dbl(lambdas, ~.[2])) %>%
    select(period, mu1, mu2, sigma1, sigma2, lambda1, lambda2)
}

# Fit a GLM to each parameter from the MIXTURE model and extract predicted values of mu, sigma, and lambda parameters
fit_glm_to_age_by_h1yy <- function(DATA) {
  
  pad <- data.frame(period = seq(2009, 2015, 1))
  
  DF <- DATA %>%
    full_join(pad, by="period") %>%
    arrange(period) %>% 
    fill(mu1, mu2, sigma1, sigma2, lambda1, lambda2) %>%
    gather("param", "value", c("mu1", "mu2", "sigma1", "sigma2", "lambda1", "lambda2")) %>%
    arrange(param, period) %>%
    group_by(param) %>%
    nest
  
  predict_it <- function(model) {
    data.frame(period = seq(2009,2030), pred = predict(model, type="response", newdata = data.frame(period = seq(2009,2030))))
  }
  
  DF2 <- DF %>%
    mutate(glm_model = map(data, ~glm(value ~ period, data = .)),
           glm_predict = map(glm_model, predict_it)) %>%
    select(param, glm_predict) %>%
    unnest(cols = glm_predict)
  
  colnames(DF2)[colnames(DF2)=="period"] <- "H1YY"
  
  return(DF2)
}

test <- test %>% 
  mutate(ini1 = pmap(list(naaccord, group), get_age_by_h1yy))

test1 <- unnest(test, ini1) %>% select(-'naaccord')
write_csv(test1, filePath(validation_dir, 'age_by_h1yy_raw.csv'))

age_by_h1yy <- test %>%
  mutate(age_by_h1yy = map(ini1, fit_glm_to_age_by_h1yy)) %>%
  select(c(group, age_by_h1yy)) %>%
  unnest(cols = age_by_h1yy) %>%
  rename_all(tolower)

write_csv(age_by_h1yy, filePath(param_dir, 'age_by_h1yy.csv'))
