######################################################################################
# Load packages
######################################################################################

suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(tidyverse))

wd <- getwd()
paramwd <- paste0(wd, "/../../data/param")
naaccordwd <- paste0(wd, "/../../data/input")

######################################################################################
#' Use to filter the dataset to the risk/race groups of interest
#' DF = name of data frame
#' VALUE = value(s) of group variable to filter on
######################################################################################
filterfx <- function(DF, VALUE) {
  filtered <- DF %>%
    filter(group %in% VALUE)
}

######################################################################################
#' Predict the number of new HAART initiators in the US, 2009-2030
#' ** updated for IDU - use this version for MSM as well
######################################################################################
surv_fx1 <- function(DF) {
  
  ##########################################
  # Shorten race coding
  ##########################################
  surv <- surv %>%
    mutate(race = gsub(' [A-z-]*', '' , race)) %>%
    # Drop "Other" race/ethnicity group AND risk
    filter(race !="Other",
           risk != "other")
  
  ##########################################
  # 02/08/19 updates
  ##########################################
  surv <- surv %>%
    mutate(race = replace(race, race=="Hispanic", "hisp"),
           group = paste0(tolower(risk), "_", tolower(race)))
  
  # Filter to groups of interest
  surv <- filterfx(surv, filtergroup)
  
  # Aggregate by group, sex, and year
  surv <- surv %>%
    group_by(group, sex, year) %>%
    summarise(n_dx = sum(n_hivdx_cdctable1)) %>%
    ungroup %>%
    select(year, group, sex, n_dx)
  
  return(surv)
}

surv_fx2 <- function(DF) {
  surv <- DF
  
  ####################################################
  # Define model formulas and helper functions
  ####################################################
  poisson_model <- function(df) {
    glm(n_dx ~ year, data=df, family="poisson")
  }
  
  gamma_model <- function(df) {
    glm(n_dx ~ year, data=df, family="Gamma")
  }
  
  ns2_model <- function(df) {
    lm(n_dx ~ ns(year, 2), data=df)
  }
  
  predict_it <- function(x,y) {
    preds2 <- data.frame(year=seq(2009,2030), pred = predict(y, type="response", newdata=x, se.fit = T))
    
    preds2 <- preds2 %>%
      mutate(lower = pred.fit - 1.96*pred.se.fit,
             upper = pred.fit + 1.96*pred.se.fit)
  }
  
  ####################################################
  # Create expanded dataset to 2030 to use for predictions
  ####################################################
  groups <- unique(surv$group)
  years <- seq(2009, 2030)
  sex <- c("Males", "Females")
  
  simulated <- expand.grid(year = years, group = groups, sex = sex, stringsAsFactors = F)
  remove <- simulated %>% filter(grepl("msm", group), sex=="Females")
  
  simulated <- simulated %>%
    anti_join(remove, by = c("group", "sex")) %>% 
    group_by(group, sex) %>% 
    nest %>%
    rename(new_data = data)

  ##########################
  # Nest the CDC data
  ##########################
  by_risk <- surv %>%
    mutate(n_dx = replace(n_dx, !is.na(n_dx), floor(n_dx))) %>%
    group_by(group, sex) %>%
    nest
  
  # Add the new data as a column
  by_risk <- by_risk %>%
    left_join(simulated, by = c("group", "sex"))
  
  ##########################
  # Poisson Model
  ##########################
  by_risk4 <- by_risk %>% 
    mutate(model = map(data, poisson_model),
           preds2 = map2(new_data, model, predict_it))
  
  fit4 <- by_risk4 %>%
    unnest(preds2) %>%
    mutate(model = "Poisson")

  ##########################
  # Gamma Model
  ##########################
  by_risk5 <- by_risk %>% 
    mutate(model = map(data, gamma_model),
           preds2 = map2(new_data, model, predict_it))
  
  fit5 <- by_risk5 %>%
    unnest(preds2) %>%
    mutate(model = "Gamma")
  
  ##########################
  # NS2 Model
  ##########################
  by_risk6 <- by_risk %>% 
    #filter(group=="idu_white") %>%
    mutate(model = map(data, ns2_model),
           preds2 = map2(new_data, model, predict_it))
  
  fit6 <- by_risk6 %>%
    unnest(preds2) %>%
    mutate(model = "NS 2")
  
  #################################################
  # Combine all models **
  #################################################
  fit_all <- bind_rows(fit4, fit5, fit6)
  
  # ID any models with negative predictions
  #negative <- fit_all %>%
  #  filter(pred.fit < 0) %>%
  #  select(group, sex, model) %>%
  #  distinct
  
  #fit_all <- fit_all %>%
  #  anti_join(negative, by=c("group", "sex", "model"))
  
  # NEW 11/05/18: Set negative predictions to 1
  fit_all <- fit_all %>%
    mutate(lower = replace(lower, lower <= 0, 1),
           upper = replace(upper, upper <= 0, 1))
  
  #################################################
  #' Create predicted #s of new Dx's
  #' IDU: use gamma, poisson, and NS2 models
  #' MSM: use gamma, poisson models
  #################################################
  remove <- fit_all %>%
    filter(grepl("msm", group), model %in% c("NS 2"))
  
  fit_all <- fit_all %>%
    anti_join(remove, by=c("group", "sex", "model"))
  
  fit_all2a <- fit_all %>%
    arrange(group, sex, year, lower) %>%
    group_by(group, sex, year) %>%
    filter(row_number()==1) %>%
    ungroup %>%
    select(group, sex, year, lower)
  
  fit_all2b <- fit_all %>%
    arrange(group, sex, year, desc(upper)) %>%
    group_by(group, sex, year) %>%
    filter(row_number()==1) %>%
    ungroup %>%
    select(group, sex, year, upper)
  
  fit_all2 <- fit_all2a %>%
    full_join(fit_all2b, by=c("group", "sex", "year"))
  
  return(fit_all2)
}

######################################################################################
# Define group(s) of interest - if multiple, separate by commas
######################################################################################
filtergroup <- c("idu_black", "idu_hisp", "idu_white",
                 "het_black", "het_hisp", "het_white",
                 "msm_black", "msm_hisp", "msm_white")

######################################################################################
# 2009 - 2030 population of new HIV Dx's - TIME-FIXED COMPONENTS
######################################################################################
# Read in dataset of # of new HIV diagnoses, 2009-2015 reported by CDC in Table 1
setwd(naaccordwd)
surv <- read.csv("dx_estimates_cdc_table1.csv", stringsAsFactors = FALSE)
colnames(surv) <- tolower(colnames(surv))

new_dx <- surv_fx1(surv)
dx_interval <- surv_fx2(new_dx)

save(new_dx, dx_interval, file=paste0(paramwd, '/dx_interval.rda'))
