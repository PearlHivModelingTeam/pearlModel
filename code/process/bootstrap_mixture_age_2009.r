rm(list = ls())
cat("\014")

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

today <- format(Sys.Date(), format="%y%m%d")

wd <- getwd()
inwd <- paste0(wd, "/../../data/input")
outwd <- paste0(wd, "/../../data/param")
######################################################################################
#' DEFINE FUNCTIONS
######################################################################################
# 1. Get population of NA-ACCORD particiapnts alive in 2009 and (as of 04/24/19) in care
fx1 <- function(DF) {
  popu2 <- DF %>%
    mutate(age2009 = 2009-yob,
           startYY = year(obs_entry),
           stopYY = year(obs_exit),
           H1YY = ifelse(year(haart1date) < 2000, 2000, year(haart1date))) %>%
    filter(startYY <= 2009, 2009 <= stopYY)
  
  # Updated 04/24/19
  carestat <- read_sas(paste0(inwd, "/popu16_carestatus.sas7bdat"))
  colnames(carestat) <- tolower(colnames(carestat))
  
  carestat <- carestat %>%
    filter(year==2009, in_care==1)
  
  popu2 <- popu2 %>%
    semi_join(carestat, by = c("naid"))
}

######################################################################################
# Read in NA-ACCORD population file and CDC Surveillance Estimates of PLWH in 2009
######################################################################################
# Popu (NA-ACCORD population created in popu.sas)
popu <- read_sas(paste0(inwd, "/popu16.sas7bdat"))
colnames(popu) <- tolower(colnames(popu))

# Format SAS dates as R dates and filter to males and females
popu <- popu %>%
  mutate(obs_entry = as.Date(obs_entry, origin="1960-01-01"),
         obs_exit = as.Date(obs_exit, origin="1960-01-01"),
         sex = replace(sex, sex==1, "Males"),
         sex = replace(sex, sex==2, "Females"),
         cd4n = replace(cd4n, cd4n < 0, NA)) %>%
  filter(sex %in% c("Males", "Females"), pop2!="") %>%
  rename(group = pop2)

popu$group <- tolower(popu$group)

# CHNAGE 09/13/18 - RECODE STUDY EXIT TO BE BASED ON MIN(DEATHDATE, COHORT_CLOSEDATE)
popu <- popu %>%
  mutate(obs_exit2 = pmin(deathdate, cohort_closedate, na.rm = T)) %>%
  select(-obs_exit) %>%
  rename(obs_exit = obs_exit2)

# CHANGE 10/30/18 - SET DEATHDATE TO MISSING IF REASON FOR EXIT IS NOT DEATH
popu <- popu %>%
  mutate(deathdate = replace(deathdate, deathdate!=obs_exit, NA))

# Nest by group and sex
popu <- popu %>%
  group_by(group, sex) %>%
  nest(.key = "data_popu")

######################################################################################
#' Get population of NA-ACCORD particiapnts alive in 2009 AND (as of 04/24/19) 
#' IN CARE IN NA-ACCORD
######################################################################################
test <- popu %>%
  mutate(naaccord_2009 = map(data_popu, fx1))

######################################################################################
#' Separate out IDU hispanic females from everyone else
######################################################################################
test_idu_hf <- test %>%
  filter(group=="idu_hisp", sex=="Females")

test <- test %>%
  anti_join(test_idu_hf, by = c("group", "sex")) %>%
  arrange(group, sex)

######################################################################################
#' BOOTSTRAP - IDU hispanic females - normal distribution
######################################################################################
test_idu_hf <- test_idu_hf %>%
  select(group, sex, naaccord_2009) %>%
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
  mutate(group = "idu_hisp",
         sex = "Females",
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
  select(group, sex, ci) %>%
  unnest

# Re-combine with IDU Hispanic Females
age2009_mixture_ci <- age2009_mixture_ci %>%
  bind_rows(idu_hf_params)

# Save
setwd(outwd)
save(age2009_mixture_ci, file = "age2009_mixture_ci.rda")

##########################################################################################
# END: IGNORE EVERYTHING BELOW THIS LINE
##########################################################################################

##########################################################################################
# QA: white females
##########################################################################################
chk <- test %>%
  slice(5) %>%
  select(naaccord_2009) %>%
  unnest

ggplot(data = chk, aes(x = age2009)) +
  geom_density()

sd(chk$age2009)
mean(chk$age2009)

B = 1000 # Number of bootstrap samples
n = length(chk)
mu1b <- mu2b <- sigma1b <- sigma2b <- lambdab <- vector(length = B)

# Bootstrap
for(i in 1:B){
  print(i)
  set.seed(i)
  dat1 = sample(chk$age2009,rep=T)
  #normalmix <- normalmixEM(dat1, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 10000)
  normalmix <- normalmixEM(dat1, k = 2, mu = c(45,50), sigma = c(5,5), lambda = c(0.1,0.9), maxit = 10000, maxrestarts = 1000)
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
 
# Seed 355 produces an error
set.seed(355)
dat1 = sample(chk$age2009,rep=T)
normalmix <- normalmixEM(dat1, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 10000)
normalmix <- normalmixEM(dat1, k = 2, mu = c(50,50), sigma = c(5,5), lambda = c(0.9, 0.1), maxit = 10000)

normalmix <- normalmixEM(dat1, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 10000,
                         epsilon = 1e-3, maxrestarts = 100)


,
                         epsilon = 1e-03)

start.par <- mean(dat1, na.rm = TRUE) + sd(dat1, na.rm = TRUE) * runif(2)

##########################################################################################
# QA: worry that the 1 pop gets swapped with the 2 pop
##########################################################################################
chk <- test %>%
  select(group, sex, mixture_params) %>%
  unnest %>%
  group_by(group, sex) %>%
  mutate(rownum = row_number()) %>%
  ungroup

chk2 <- chk %>%
  filter(mu2 < mu1)

chk2 

chk2 <- chk2 %>%
  mutate(lambda_rev = 1 - lambda1) %>%
  select(group, sex, mu2, mu1, sigma2, sigma1, lambda_rev, rownum)

colnames(chk2) <- c("group", "sex", "mu1", "mu2", "sigma1", "sigma2", "lambda1", "rownum")

chk <- chk %>%
  anti_join(chk2, by = c("group", "sex", "rownum")) %>%
  bind_rows(chk2) %>%
  select(-rownum)




##########################################################################################
# IGNORE
##########################################################################################
  ci_fx <- function(DF, VAR) {
    # Pull the CIs
    mu1 <- c(quantile(mu1b, 0.025, na.rm = T), quantile(mu1b, 0.975, na.rm = T))
    mu2 <- c(quantile(mu2b, 0.025, na.rm = T), quantile(mu2b, 0.975, na.rm = T))
    sigma1 <- c(quantile(sigma1b, 0.025, na.rm = T), quantile(sigma1b, 0.975, na.rm = T))
    sigma2 <- c(quantile(sigma2b, 0.025, na.rm = T), quantile(sigma2b, 0.975, na.rm = T))
    lambda1 <- c(quantile(lambdab, 0.025, na.rm = T), quantile(lambdab, 0.975, na.rm = T))
    
    params <- bind_rows(mu1, mu2, sigma1, sigma2, lambda1)
  }
  
# Run the bootstrapping & extract 95% CIs
ptm <- proc.time()
test <- test %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))
proc.time() - ptm


# QA: don't include non-convergent models
boot_fx <- function(DF) {
  B = 50 # Number of bootstrap samples
  n = length(DF)
  mu1b <- mu2b <- sigma1b <- sigma2b <- lambdab <- vector(length = B)
  
  # Bootstrap
  for(i in 1:B){
    print(i)
    dat1 = sample(DF$age2009,rep=T)
    normalmix <- normalmixEM(dat1, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 4000)
    if (length(normalmix$all.loglik) < 4000) {
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
  
  # Pull the CIs
  mu1 <- c(quantile(mu1b, 0.025, na.rm = T), quantile(mu1b, 0.975, na.rm = T))
  mu2 <- c(quantile(mu2b, 0.025, na.rm = T), quantile(mu2b, 0.975, na.rm = T))
  sigma1 <- c(quantile(sigma1b, 0.025, na.rm = T), quantile(sigma1b, 0.975, na.rm = T))
  sigma2 <- c(quantile(sigma2b, 0.025, na.rm = T), quantile(sigma2b, 0.975, na.rm = T))
  lambda1 <- c(quantile(lambdab, 0.025, na.rm = T), quantile(lambdab, 0.975, na.rm = T))
  
  params <- bind_rows(mu1, mu2, sigma1, sigma2, lambda1)
  
}

chk0 <- test %>%
  slice(4) %>%
  mutate(mixture_params = map(naaccord_2009, boot_fx))

chk0 <- chk0 %>%
  select(mixture_params) %>% 
  unnest

# Compare NORMALMIXEM VS NORMALMIXEM2COMP
chk <- test %>%
  slice(12) %>%
  select(group, sex, naaccord_2009) %>%
  unnest

set.seed(1)
mix1 <- normalmixEM(chk$age2009, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 4000)

set.seed(1)
mix2 <- normalmixEM2comp(chk$age2009, lambda = 0.5, mu = c(30, 50), sigsqrd = c(1,1), maxit = 4000)

# Compare the 2 methods: timed
## Run 1
B = 50 # Number of bootstrap samples
n = length(chk)
mu1b <- mu2b <- sigma1b <- sigma2b <- lambdab <- vector(length = B)

# Bootstrap
ptm <- proc.time()
for(i in 1:B){
  print(i)
  #set.seed(i)
  dat1 = sample(chk$age2009,rep=T)
  normalmix <- normalmixEM(dat1, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 10000)
  
    mu1b[i]    = normalmix$mu[1]      
    mu2b[i]    = normalmix$mu[2]      
    sigma1b[i] = normalmix$sigma[1]   
    sigma2b[i] = normalmix$sigma[2]   
    lambdab[i] = normalmix$lambda[1]
} 
# Pull the CIs
mu1 <- c(quantile(mu1b, 0.025, na.rm = T), quantile(mu1b, 0.975, na.rm = T))
mu2 <- c(quantile(mu2b, 0.025, na.rm = T), quantile(mu2b, 0.975, na.rm = T))
sigma1 <- c(quantile(sigma1b, 0.025, na.rm = T), quantile(sigma1b, 0.975, na.rm = T))
sigma2 <- c(quantile(sigma2b, 0.025, na.rm = T), quantile(sigma2b, 0.975, na.rm = T))
lambda1 <- c(quantile(lambdab, 0.025, na.rm = T), quantile(lambdab, 0.975, na.rm = T))

params1 <- bind_rows(mu1, mu2, sigma1, sigma2, lambda1)
proc.time() - ptm

## Run 2 - not as reliable
B = 50 # Number of bootstrap samples
n = length(chk)
mu1b2 <- mu2b2 <- sigma1b2 <- sigma2b2 <- lambdab2 <- vector(length = B)

# Bootstrap
ptm <- proc.time()
for(i in 1:B){
  print(i)
  #set.seed(i)
  dat1 = sample(chk$age2009,rep=T)
  normalmix <- normalmixEM2comp(dat1, lambda = 0.5, mu = c(30, 50), sigsqrd = c(1,1), maxit = 10000)
  
  mu1b2[i]    = normalmix$mu[1]      
  mu2b2[i]    = normalmix$mu[2]      
  sigma1b2[i] = normalmix$sigma[1]   
  sigma2b2[i] = normalmix$sigma[2]   
  lambdab2[i] = normalmix$lambda[1]
} 
# Pull the CIs
mu1 <- c(quantile(mu1b2, 0.025, na.rm = T), quantile(mu1b2, 0.975, na.rm = T))
mu2 <- c(quantile(mu2b2, 0.025, na.rm = T), quantile(mu2b2, 0.975, na.rm = T))
sigma1 <- c(quantile(sigma1b2, 0.025, na.rm = T), quantile(sigma1b2, 0.975, na.rm = T))
sigma2 <- c(quantile(sigma2b2, 0.025, na.rm = T), quantile(sigma2b2, 0.975, na.rm = T))
lambda1 <- c(quantile(lambdab2, 0.025, na.rm = T), quantile(lambdab2, 0.975, na.rm = T))

params2 <- bind_rows(mu1, mu2, sigma1, sigma2, lambda1)
proc.time() - ptm

# Do 1000 bootstrap and compare the density plots
B = 500
n = length(chk)
mu1b <- mu2b <- sigma1b <- sigma2b <- lambdab <- vector(length = B)

for(i in 1:B){
  dat1 = sample(chk$age2009,rep=T)
  normalmix <- normalmixEM(dat1, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 10000)
  
  mu1b[i]    = normalmix$mu[1]      
  mu2b[i]    = normalmix$mu[2]      
  sigma1b[i] = normalmix$sigma[1]   
  sigma2b[i] = normalmix$sigma[2]   
  lambdab[i] = normalmix$lambda[1]
}

# Generate random ages from each mixture distribution
i <- 1
components <- sample(1:2, prob=c(lambdab[i], 1-lambdab[i]), size=100000, replace=TRUE)
mus <- c(mu1b[i], mu2b[i])
sds <- c(sigma1b[i], sigma2b[i])
sim_mixed <- data.frame(age2009 = rnorm(n=100000, mean=mus[components], sd=sds[components]))

p <- ggplot(sim_mixed, aes(x = age2009)) +
  geom_density()

for(i in 2:500) {
  components <- sample(1:2, prob=c(lambdab[i], 1-lambdab[i]), size=100000, replace=TRUE)
  mus <- c(mu1b[i], mu2b[i])
  sds <- c(sigma1b[i], sigma2b[i])
  sim_mixed <- data.frame(age2009 = rnorm(n=100000, mean=mus[components], sd=sds[components]))
  
  p <- p +
    geom_density(data = sim_mixed, aes(x = age2009))
  
}



params %>% 
  summarise_all(list(p25 = quantile), probs = 0.25, na.rm = T)

params %>% 
  summarise_all(list(p25 = ~quantile(., probs = 0.25, na.rm = T),
                     p75 = ~quantile(., probs = 0.75, na.rm = T)))
