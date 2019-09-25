######################################################################################
#' Log of changes
#' 05/07/2019: Fit a mixture distribution to NA-ACCORD age in 2009
######################################################################################

######################################################################################
# Load packages
######################################################################################
library(MASS)
library(haven)
library(lubridate)
library(mixtools)
library(openxlsx)
library(tidyverse)
library(R.utils)
library(feather)

input_dir <- filePath(getwd(), '/../../data/input')
param_dir <- filePath(getwd(), '/../../data/param')

test <- read_feather(filePath(input_dir, 'naaccord_2009.feather'))

# Nest by group 
test <- test %>%
  group_by(group) %>%
  nest(.key = "naaccord_2009")

######################################################################################
#' Separate out IDU hispanic females from everyone else
######################################################################################
test_idu_hf <- test %>%
  filter(group=="idu_hisp_female")

test <- test %>%
  anti_join(test_idu_hf, by = c("group")) %>%
  arrange(group)

######################################################################################
#' BOOTSTRAP - IDU hispanic females - normal distribution
######################################################################################
test_idu_hf <- test_idu_hf %>%
  select(group, naaccord_2009) %>%
  unnest

# Boostrap the mean and SD
B = 1000 # Number of bootstrap samples
n = length(test_idu_hf)
mub <-sigmab <- vector(length = B)

# Bootstrap
for(i in 1:B){
  set.seed(i)
  dat1 <- sample(test_idu_hf$age2009, rep=T)
  
  fit <- fitdistr(dat1, "normal")
  
  mub[i] <- fit$estimate[1]
  sigmab[i] <- fit$estimate[2] 
}

idu_hf_params <- data.frame(mu1 = mub, sigma1 = sigmab)

# Pull the quantiles
idu_hf_params <- idu_hf_params %>%
  summarise_all(list(p025 = ~quantile(., probs = 0.025, na.rm = T),
                     p975 = ~quantile(., probs = 0.975, na.rm = T)))


idu_hf_params <- idu_hf_params %>%
  mutate(group = "idu_hisp_female",
         lambda1_p025 = 1,
         lambda1_p975 = 1) 

######################################################################################
#' BOOTSTRAP - all groups except for IDU hispanic females
######################################################################################
# Bootstrap fx
boot_fx <- function(DF) {
  B = 1000 # Number of bootstrap samples
  n = length(DF)
  mu1b <- mu2b <- sigma1b <- sigma2b <- lambdab <- vector(length = B)
  
  # Bootstrap
  for(i in 1:B){
    #print(i)
    set.seed(i)
    dat1 = sample(DF$age2009,rep=T)
    normalmix <- normalmixEM(dat1, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 10000, maxrestarts = 1000)
    if (length(normalmix$all.loglik) < 10000) {
      mu1b[i]    = normalmix$mu[1]      
      mu2b[i]    = normalmix$mu[2]      
      sigma1b[i] = normalmix$sigma[1]   
      sigma2b[i] = normalmix$sigma[2]   
      lambdab[i] = normalmix$lambda[1]
    } else {
      mu1b[i] <- NA
      mu2b[i]    = NA     
      sigma1b[i] = NA
      sigma2b[i] = NA
      lambdab[i] = NA
    }
    
  }
  
  params <- data.frame(mu1 = mu1b, mu2 = mu2b, sigma1 = sigma1b, sigma2 = sigma2b, lambda1 = lambdab)
}

# Run the bootstrapping
test1 <- test %>%
  slice(1) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test2 <- test %>%
  slice(2) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test3 <- test %>%
  slice(3) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test4 <- test %>%
  slice(4) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test5 <- test %>%
  slice(5) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx)) #- white females - 589.93

test6 <- test %>%
  slice(6) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test7 <- test %>%
  slice(7) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test8 <- test %>%
  slice(8) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test9 <- test %>%
  slice(9) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test10 <- test %>%
  slice(10) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx)) ##

test11 <- test %>%
  slice(11) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test12 <- test %>%
  slice(12) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test13 <- test %>%
  slice(13) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

test14 <- test %>%
  slice(14) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

# Get the 95% CIs
ci_fx <- function(DF) {
  
  # Swap the 1 and 2 pops if the 2 pop is younger than the 1 pop
  DF1 <- DF %>%
    mutate(rownum = row_number())
  
  DF2 <- DF1 %>%
    filter(mu2 < mu1)
  
  DF2 <- DF2 %>%
    mutate(lambda_rev = 1 - lambda1) %>%
    select(mu2, mu1, sigma2, sigma1, lambda_rev, rownum)
  
  colnames(DF2) <- c("mu1", "mu2", "sigma1", "sigma2", "lambda1", "rownum")
  
  DF1 <- DF1 %>%
    anti_join(DF2, by = c("rownum")) %>%
    bind_rows(DF2) %>%
    select(-rownum)
  
  # Get the 95% CIs
  DF1 <- DF1 %>%
    summarise_all(list(p025 = ~quantile(., probs = 0.025, na.rm = T),
                       p975 = ~quantile(., probs = 0.975, na.rm = T)))
}

test0 <- test1 %>%
  bind_rows(test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13, test14)

test0 <- test0 %>%
  mutate(ci = map(mixture_params, ci_fx))

age2009_mixture_ci <- test0 %>%
  select(group, ci) %>%
  unnest

# Re-combine with IDU Hispanic Females
age2009_mixture_ci <- age2009_mixture_ci %>%
  bind_rows(idu_hf_params)

# Save
write_feather(age2009_mixture_ci, filePath(param_dir, 'age_in_2009.feather')
