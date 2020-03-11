# Load packages ---------------------------------------------------------------
suppressMessages(library(gamlss))
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(feather))
suppressMessages(library(lubridate))

input_dir <- filePath(getwd(), '/../../data/input/aim_1')
param_dir <- filePath(getwd(), '/../../data/parameters/aim_1')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')


######################################################################################
#' Predict the number of new HAART initiators in the US, 2009-2030
#' ** updated for IDU - use this version for MSM as well
######################################################################################
surv_fx1 <- function(df) {
  
  ##########################################
  # Shorten race coding
  ##########################################
  df <- df %>%
    mutate(race = gsub(' [A-z-]*', '' , race)) %>%
    # Drop "Other" race/ethnicity group AND risk
    filter(race !="Other",
           risk != "other")
  
  ##########################################
  # 02/08/19 updates
  ##########################################
  df <- df %>%
    mutate(race = replace(race, race=="Hispanic", "hisp"),
           group = paste0(tolower(risk), "_", tolower(race)),
           sex = replace(sex, sex=="Females", 'female'),
           sex = replace(sex, sex=='Males', 'male'))
  
  # Filter to groups of interest
  #df <- filterfx(df, filtergroup)
  
  # Aggregate by group, sex, and year
  df <- df %>%
    group_by(group, sex, year) %>%
    summarise(n_dx = sum(n_hivdx_cdctable1)) %>%
    ungroup %>%
    select(year, group, sex, n_dx) %>%
    unite(group, group, sex)
  
  return(df)
}

# Predict new interval of new diagnoses by year ----------------------
predict_new_dx <- function(DF) {
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
    preds2 <- data.frame(year=seq(2006,2030), pred = predict(y, type="response", newdata=x, se.fit = T))
    
    preds2 <- preds2 %>%
      mutate(lower = pred.fit - 1.96*pred.se.fit,
             upper = pred.fit + 1.96*pred.se.fit)
  }
  
  ####################################################
  # Create expanded dataset to 2030 to use for predictions
  ####################################################
  groups <- unique(surv$group)
  years <- seq(2006, 2030)
  
  simulated <- expand.grid(year = years, group = groups, stringsAsFactors = F)
  
  simulated <- simulated %>%
    group_by(group) %>% 
    nest %>%
    rename(new_data = data)
  
  ##########################
  # Nest the CDC data
  ##########################
  by_risk <- surv %>%
    mutate(n_dx = replace(n_dx, !is.na(n_dx), floor(n_dx))) %>%
    group_by(group) %>%
    nest
  
  # Add the new data as a column
  by_risk <- by_risk %>%
    left_join(simulated, by = c("group"))
  
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
    anti_join(remove, by=c("group", "model"))

  remove <- fit_all %>%
    filter(grepl("idu_black", group), model %in% c("Gamma"))

  fit_all <- fit_all %>%
    anti_join(remove, by=c("group", "model"))

  fit_all2a <- fit_all %>%
    arrange(group, year, lower) %>%
    group_by(group, year) %>%
    filter(row_number()==1) %>%
    ungroup %>%
    select(group, year, lower)
  
  fit_all2b <- fit_all %>%
    arrange(group, year, desc(upper)) %>%
    group_by(group, year) %>%
    filter(row_number()==1) %>%
    ungroup %>%
    select(group, year, upper)
  
  fit_all2 <- fit_all2a %>%
    full_join(fit_all2b, by=c("group", "year"))
  
  return(fit_all2)
}

cdc_estimates <- read.csv(filePath(input_dir, 'dx_estimates_cdc_table1.csv'), stringsAsFactors = FALSE) %>%
  rename_all(tolower)

new_dx <- surv_fx1(cdc_estimates)
#write_feather(new_dx, filePath(param_dir, 'new_dx.feather'))

new_dx_interval <- predict_new_dx(new_dx)
write_feather(new_dx_interval, filePath(param_dir, 'new_dx_interval.feather'))
