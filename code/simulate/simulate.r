######################################################################################
# Load packages
######################################################################################
suppressPackageStartupMessages(library(argparse))
parser <- ArgumentParser() 
parser$add_argument("-p", "--parallel", type="logical", default=FALSE,
                    help="Run parallel? default is FALSE")
parser$add_argument("-c", "--cores", type="integer", default=0, 
                    help="Number of cores to use, default is (total cores -1)") 
parser$add_argument("-r", "--replications", type="integer", default=100, 
                    help="Number of replications, default is 100") 
parser$add_argument("-f", "--filter", type="logical", default=FALSE, 
                    help="Filter to het_black? default: FALSE") 
args <- parser$parse_args()

library(MASS)
library(haven)
suppressMessages(library(lubridate))
library(broom)
library(geepack)
suppressMessages(library(mixtools))
library(RGeode)
library(triangle)
suppressMessages(library(gamlss))
library(binom)
suppressMessages(library(tidyverse))

cwd <- getwd()
file <- paste0(cwd, '/../../data/processed/processed.rda')
load(file)

rep <- 1

######################################################################################
# Define group(s) of interest - if multiple, separate by commas
######################################################################################
filtergroup <- c("idu_black", "idu_hisp", "idu_white",
                 "het_black", "het_hisp", "het_white",
                 "msm_black", "msm_hisp", "msm_white")
if(args$filter) {filtergroup <- c('het_black')}

today <- format(Sys.Date(), format="%y%m%d")

######################################################################################
# Time-varying components
######################################################################################
wrapper <- function(rep, groupname, sexvalue, prob_reengage, linelist) {
  # Temporarily run on a subset of groups
  test <- filterfx(test, groupname)
  
  test <- test %>%
    filter(sex==sexvalue)
  
  #########################
  # 2009 population of PLWH
  #########################
  # Decide between weibull and mixture and simulate final 2009 population - REVISED 10/24/18 TO DROP WEIBULL_2009
  test <- test %>%
    mutate(art2009 = pmap(list(mixture_2009, naaccord_2009, on_art, naaccord_prop_2009, naaccord_cd4_2009), fx6))
  
  #######################################
  # 2009-2015 population of new initiators
  #######################################
  # INI - # of new HAART initiators in the US, 2009-2030
  test <- test %>%
    mutate(data_ini = map2(hiv_pred_interval, surv, surv_fx3))
  
  # Clean the mixed normal parameter estimates and sample the param estimate values
  test <- test %>%
    mutate(ini3 = map(ini2, inifx3))
  
  # 6. Simulate the population (mixed normal)
  test <- test %>%
    mutate(new_art_age_mixed = pmap(list(data_ini, ini3), inifx4))
  
  # 7. Now simulate CD4 count at HAART initiation (modified inputs for IDU)
  test <- test %>%
    mutate(new_art_mixture = pmap(list(data_popu, group, sex, new_art_age_mixed), ini_cd4_fx))
  
  ######################################################################################
  ######################################################################################
  # PART 2: Estimate age distribution of simulated population, 2009-2030
  ######################################################################################
  ######################################################################################
  
  #############################
  # 2009 starting pop
  #############################
  art2009 <- test %>% 
    unnest(art2009)
  
  colnames(art2009) <- tolower(colnames(art2009))
  
  # Filter to population of patients alive in 2009 and on HAART
  art2009 <- art2009 %>%
    mutate(calyy = 2009) %>%
    select(group, sex, age, h1date, h1yy, sqrtcd4n, intercept_est, ageby10_est, sqrtcd4n_est, year_est, h1yy_est, calyy)
  
  # Create some new variables
  art2009 <- art2009 %>%
    mutate(patid = row_number(),
           agecat = floor(age/10),
           agecat = replace(agecat, agecat < 2, 2),
           agecat = replace(agecat, agecat > 7, 7))
  
  # Create time-varying CD4
  keep_names <- colnames(art2009)
  
  art2009 <- art2009 %>%
    left_join(coeff_cd4_increase, by = c("group", "sex")) %>%
    mutate(cd4cat = case_when(sqrtcd4n < sqrt(200) ~ 1,
                              sqrtcd4n < sqrt(350) ~ 2,
                              sqrtcd4n < sqrt(500) ~ 3,
                              sqrtcd4n >= sqrt(500) ~ 4),
           cd4cat349 = as.numeric(cd4cat==2),
           cd4cat499 = as.numeric(cd4cat==3),
           cd4cat500 = as.numeric(cd4cat==4)) %>%
    mutate(time_from_h1yy = calyy - h1yy,
           time_from_h1yy_ = (pmax(0, time_from_h1yy-p5)^2 - pmax(0, time_from_h1yy-p95)^2) / (p95-p5),
           time_from_h1yy__  = (pmax(0, time_from_h1yy-p35)^2 - pmax(0, time_from_h1yy-p95)^2) / (p95-p5),
           time_from_h1yy___  = (pmax(0, time_from_h1yy-p65)^2 - pmax(0, time_from_h1yy-p95)^2) / (p95-p5)) %>%
    mutate(tv_sqrtcd4n = intercept_c + 
             (agecat_c*agecat) + 
             (cd4cat349_c*cd4cat349) + (cd4cat499_c*cd4cat499) + (cd4cat500_c*cd4cat500) +
             (time_from_h1yy_c*time_from_h1yy) +
             (`_time_from_h1yy_c`*time_from_h1yy_) +
             (`__time_from_h1yy_c`*time_from_h1yy__) +
             (`___time_from_h1yy_c`*time_from_h1yy___) +
             (`_time_from_cd4cat349_c`*time_from_h1yy_*cd4cat349) +
             (`_time_from_cd4cat499_c`*time_from_h1yy_*cd4cat499) +
             (`_time_from_cd4cat500_c`*time_from_h1yy_*cd4cat500) +
             (`__time_fro_cd4cat349_c`*time_from_h1yy__*cd4cat349) +
             (`__time_fro_cd4cat499_c`*time_from_h1yy__*cd4cat499) +
             (`__time_fro_cd4cat500_c`*time_from_h1yy__*cd4cat500) +
             (`___time_fr_cd4cat349_c`*time_from_h1yy___*cd4cat349) +
             (`___time_fr_cd4cat499_c`*time_from_h1yy___*cd4cat499) +
             (`___time_fr_cd4cat500_c`*time_from_h1yy___*cd4cat500)) %>%
    mutate(status = 0) %>%
    select(keep_names, tv_sqrtcd4n, status)
  
  # Get n by age group within each sex and risk group
  transed <- art2009 %>%
    group_by(group, sex, agecat) %>%
    tally %>%
    ungroup %>%
    spread(agecat, n) %>%
    mutate(calyy = 2009)
  
  # Initilize a list and store the 2009 age data 
  pcntdat <- vector("list", 22)
  names(pcntdat) <- paste0("yr", seq(2009,2030))
  pcntdat[1] <- list(transed) 
  
  # Get the baseline characteristics
  fit_stats09 <- art2009 %>%
    group_by(group, sex, calyy) %>%
    summarise(n_alive = n(),
              n_ini = 0,
              mean_age = mean(age),
              median_age = median(age),
              p25_age = quantile(age, probs = 0.25),
              p75_age = quantile(age, probs = 0.75),
              mean_sqrtcd4n = mean(sqrtcd4n)) %>%
    ungroup
  
  all_stats <- vector("list", 22)
  names(all_stats) <- paste0("yr", seq(2009,2030))
  all_stats[1] <- list(fit_stats09) 
  
  #############################
  # 2010-2030 new ART init date
  #############################
  newart <- test %>%
    select(group, sex, new_art_mixture, ends_with("_est")) %>%
    unnest %>%
    select(group, sex, age = iniage, h1date, h1yy = H1YY, sqrtcd4n, intercept_est, ageby10_est, sqrtcd4n_est, year_est, h1yy_est) %>%
    mutate(calyy = h1yy,
           patid = row_number()+nrow(art2009),
           agecat = floor(age/10),
           agecat=replace(agecat, agecat < 2, 2),
           agecat=replace(agecat, agecat > 7, 7),
           tv_sqrtcd4n = sqrtcd4n,
           status = 0)
  
  # Get the # of new INIs by year, group, and sex to save for the output report
  newart_n <- newart %>% group_by(group, sex, h1yy) %>% tally %>% ungroup
  
  # Tally by individual age year
  newart_age <- newart %>% group_by(group, sex, age, h1yy) %>% tally %>% ungroup
  
  #############################
  # Initiate lists to store output
  #############################
  in_care <- vector("list", 22)
  names(in_care) <- paste0("yr", seq(2009,2030))
  
  dead_care <- vector("list", 22)
  names(dead_care) <- paste0("yr", seq(2009,2030))
  
  newly_lost <- vector("list", 22)
  names(newly_lost) <- paste0("yr", seq(2009,2030))
  
  out_care <- vector("list", 22)
  names(out_care) <- paste0("yr", seq(2009,2030))
  
  dead_out_care <- vector("list", 22)
  names(dead_out_care) <- paste0("yr", seq(2009,2030))
  
  newly_reengage <- vector("list", 22)
  names(newly_reengage) <- paste0("yr", seq(2009,2030))
  
  #############################
  # Allow LTF to occur in 2009
  #############################
  # Calculate the probability of loss to follow-up among those who did not die
  pop1 <- art2009
  keep_names <- colnames(pop1)
  
  pop1 <- pop1 %>%
    left_join(pctls_leave_na, by = c("group", "sex")) %>%
    mutate(age_ = (pmax(0, age-p5)^2 - pmax(0, age-p95)^2) / (p95-p5),
           age__  = (pmax(0, age-p35)^2 - pmax(0, age-p95)^2) / (p95-p5),
           age___  = (pmax(0, age-p65)^2 - pmax(0, age-p95)^2) / (p95-p5),
           haart_period = as.numeric(h1yy > 2010)) %>%
    left_join(coeff_leave_na, by = c("group", "sex")) %>%
    mutate(y = intercept_c + (age_c*age) + (`_age_c`*age_) + (`__age_c`*age__) + (`___age_c`*age___) + (year_c*calyy) + (sqrtcd4n_c*sqrtcd4n) + (haart_period_c*haart_period),
           prob = exp(y) / (1 + exp(y)),
           rr = runif(nrow(pop1)),
           status = 0,
           status = replace(status, rr <= prob, 2)) %>%
    select(keep_names)
  
  # Separate out the people LTFU
  ltfu1 <- pop1 %>% filter(status==2) %>% mutate(ltfu_year = 2009, sqrtcd4n_exit = tv_sqrtcd4n)
  pop1 <- pop1 %>% filter(status==0)
  
  # Output people to their proper lists
  in_care[1] <- list(pop1)
  newly_lost[1] <- list(ltfu1)
  
  #############################
  # Define years to loop through
  #############################
  seqyears <- seq(2010,2030)
  
  #############################
  # Loop through years to get predicted age distribution
  #############################
  for (YY in seq_along(seqyears)) {
    
    # Define placeholder for previous year
    preYY <- seqyears[YY] - 1
    
    #################################
    # POP1: IN CARE
    #################################
    # Get the datafile of patients who survived from the PREVIOUS year
    pop1 <- in_care[[paste0("yr", preYY)]] 
    
    # Add in persons who re-engaged in care at the end of the previous year
    comeback <- newly_reengage[[paste0("yr", preYY)]] 
    
    pop1 <- bind_rows(pop1, comeback)
    pop1 <- pop1[ , !(names(pop1) %in% c("ltfu_year"))]
    
    # Allow the patients from the previous year to age 1 year
    pop1 <- pop1 %>%
      mutate(age = age + 1,
             calyy = calyy + 1) %>%
      mutate(agecat = floor(age/10)) %>%
      mutate(agecat=replace(agecat, agecat < 2, 2)) %>%
      mutate(agecat=replace(agecat, agecat > 7, 7)) 
    
    # Simulate new sqrt CD4
    keep_names <- colnames(pop1)
    
    pop1$tv_sqrtcd4n <- NA # Set CD4 values from the previous year to missing so that we can re-generate them
    
    pop1 <- pop1 %>%
      left_join(coeff_cd4_increase, by = c("group", "sex")) %>%
      mutate(cd4cat = case_when(sqrtcd4n < sqrt(200) ~ 1,
                                sqrtcd4n < sqrt(350) ~ 2,
                                sqrtcd4n < sqrt(500) ~ 3,
                                sqrtcd4n >= sqrt(500) ~ 4),
             cd4cat349 = as.numeric(cd4cat==2),
             cd4cat499 = as.numeric(cd4cat==3),
             cd4cat500 = as.numeric(cd4cat==4)) %>%
      mutate(time_from_h1yy = calyy - h1yy,
             time_from_h1yy_ = (pmax(0, time_from_h1yy-p5)^2 - pmax(0, time_from_h1yy-p95)^2) / (p95-p5),
             time_from_h1yy__  = (pmax(0, time_from_h1yy-p35)^2 - pmax(0, time_from_h1yy-p95)^2) / (p95-p5),
             time_from_h1yy___  = (pmax(0, time_from_h1yy-p65)^2 - pmax(0, time_from_h1yy-p95)^2) / (p95-p5)) %>%
      mutate(tv_sqrtcd4n = intercept_c + 
               (agecat_c*agecat) + 
               (cd4cat349_c*cd4cat349) + (cd4cat499_c*cd4cat499) + (cd4cat500_c*cd4cat500) +
               (time_from_h1yy_c*time_from_h1yy) +
               (`_time_from_h1yy_c`*time_from_h1yy_) +
               (`__time_from_h1yy_c`*time_from_h1yy__) +
               (`___time_from_h1yy_c`*time_from_h1yy___) +
               (`_time_from_cd4cat349_c`*time_from_h1yy_*cd4cat349) +
               (`_time_from_cd4cat499_c`*time_from_h1yy_*cd4cat499) +
               (`_time_from_cd4cat500_c`*time_from_h1yy_*cd4cat500) +
               (`__time_fro_cd4cat349_c`*time_from_h1yy__*cd4cat349) +
               (`__time_fro_cd4cat499_c`*time_from_h1yy__*cd4cat499) +
               (`__time_fro_cd4cat500_c`*time_from_h1yy__*cd4cat500) +
               (`___time_fr_cd4cat349_c`*time_from_h1yy___*cd4cat349) +
               (`___time_fr_cd4cat499_c`*time_from_h1yy___*cd4cat499) +
               (`___time_fr_cd4cat500_c`*time_from_h1yy___*cd4cat500)) %>%
      select(keep_names, tv_sqrtcd4n)
    
    # Add in the new HAART initiators
    newARTinOneYearData <- newart %>%
      filter(h1yy == seqyears[YY])
    
    pop1 <- bind_rows(pop1, newARTinOneYearData)
    
    # Calculate probability of death
    pop1 <- pop1 %>%
      mutate(status = 0,
             y = intercept_est + (ageby10_est*agecat) + (sqrtcd4n_est*sqrtcd4n) + (year_est*calyy) + (h1yy_est*h1yy),
             prob = exp(y) / (1 + exp(y)),
             rr = runif(nrow(pop1))) %>%
      mutate(status = replace(status, rr <= prob, 1)) %>%
      mutate(status = replace(status, age > 85, 1)) %>%
      select(-y, -prob, -rr)
    
    # Separate the people who died
    dead1 <- pop1 %>% filter(status==1)
    
    pop1 <- pop1 %>% filter(status==0)
    
    # Calculate the probability of loss to follow-up among those who did not die
    keep_names <- colnames(pop1)
    
    pop1 <- pop1 %>%
      left_join(pctls_leave_na, by = c("group", "sex")) %>%
      mutate(age_ = (pmax(0, age-p5)^2 - pmax(0, age-p95)^2) / (p95-p5),
             age__  = (pmax(0, age-p35)^2 - pmax(0, age-p95)^2) / (p95-p5),
             age___  = (pmax(0, age-p65)^2 - pmax(0, age-p95)^2) / (p95-p5),
             haart_period = as.numeric(h1yy > 2010)) %>%
      left_join(coeff_leave_na, by = c("group", "sex")) %>%
      mutate(y = intercept_c + (age_c*age) + (`_age_c`*age_) + (`__age_c`*age__) + (`___age_c`*age___) + (year_c*calyy) + (sqrtcd4n_c*sqrtcd4n) + (haart_period_c*haart_period),
             prob = exp(y) / (1 + exp(y)),
             rr = runif(nrow(pop1)),
             status = 0,
             status = replace(status, rr <= prob, 2)) %>%
      select(keep_names)
    
    # Separate out the people LTFU
    ltfu1 <- pop1 %>% filter(status==2) %>% mutate(ltfu_year = seqyears[YY], sqrtcd4n_exit = tv_sqrtcd4n)
    pop1 <- pop1 %>% filter(status==0)
    
    # Output people to their proper lists (add 1 because the lists all start in 2009 not 2010)
    dead_care[YY+1] <- list(dead1)
    in_care[YY+1] <- list(pop1)
    newly_lost[YY+1] <- list(ltfu1)
    
    #################################
    # POP2: OUT OF CARE
    #################################
    keep_names_final <- c("group", "sex", "age", "h1date", "h1yy", "sqrtcd4n", "intercept_est", "ageby10_est", "sqrtcd4n_est", "year_est", 
                          "h1yy_est", "calyy", "patid", "agecat", "tv_sqrtcd4n", "status", "ltfu_year", "sqrtcd4n_exit")
    
    # Get the datafile of patients who remained lost to follow up at the end of the previous year
    pop2 <- out_care[[paste0("yr", preYY)]] 
    
    # Add the patients who were newly lost to follow up at the end of the previous year
    pop2b <- newly_lost[[paste0("yr", preYY)]] 
    
    pop2 <- bind_rows(pop2, pop2b)
    
    if (nrow(pop2 >= 1)) {
      # Allow the patients to age 1 year
      pop2 <- pop2 %>%
        mutate(age = age + 1,
               calyy = calyy + 1) %>%
        mutate(agecat = floor(age/10)) %>%
        mutate(agecat=replace(agecat, agecat < 2, 2)) %>%
        mutate(agecat=replace(agecat, agecat > 7, 7))
      
      # Simulate CD4 decline
      keep_vars <- colnames(pop2)
      
      pop2 <- pop2 %>%
        mutate(time_out = calyy - ltfu_year) %>%
        mutate(diff = coeff_cd4_decrease$intercept_c + (coeff_cd4_decrease$time_out_of_naaccord_c*time_out) + (coeff_cd4_decrease$sqrtcd4_exit_c*sqrtcd4n_exit),
               tv_sqrtcd4n = sqrt((sqrtcd4n_exit**2)*exp(diff)*1.5)) %>%
        select(keep_vars)
      
      # Calculate probability of death
      keep_vars <- colnames(pop2)
      
      pop2 <- pop2 %>%
        left_join(coeff_mortality_out, by = c("group", "sex")) %>%
        mutate(y = intercept_c + (year_c*calyy) + (agecat_c*agecat) + (tv_sqrtcd4n_c*tv_sqrtcd4n),
               prob = exp(y) / (1 + exp(y)),
               rr = runif(nrow(pop2)),
               status = 2) %>%
        mutate(status = replace(status, rr <= prob, -1)) %>%
        mutate(status = replace(status, age > 85, -1)) %>%
        select(keep_vars)
      
      # Separate the people who died
      dead2 <- pop2 %>% filter(status==-1)
      
      pop2 <- pop2 %>% filter(status==2)
      
      # Evaluate the probability of re-engaging
      
      pop2 <- pop2 %>%
        mutate(prob = prob_reengage,
               rr = runif(nrow(pop2))) %>%
        mutate(status = replace(status, rr <= prob, 3)) %>%
        select(keep_names_final)
      
      # Separate the people who re-engaged
      ltfu2 <- pop2 %>% filter(status==3) %>% select(colnames(pop1), ltfu_year)
      pop2 <- pop2 %>% filter(status==2)
      
    } else {
      dead2 <- dead1[FALSE,]
      pop2 <- ltfu1[FALSE,]
      ltfu2 <- pop1[FALSE,] %>% mutate(ltfu_year = NA)
    }
    
    # Output people to their proper lists
    dead_out_care[YY+1] <- list(dead2)
    out_care[YY+1] <- list(pop2)
    newly_reengage[YY+1] <- list(ltfu2)
    
  }
  
  ############################################
  # linelist for Parastu
  ############################################
  if (linelist==1) {
    myvars <- c("patid", "group", "sex", "h1yy", "sqrtcd4n", "calyy", "age", "tv_sqrtcd4n", "status")
    
    dead_out_care_r <-  bind_rows(dead_out_care) %>%
      select(myvars)
    
    dead_care_r <-  bind_rows(dead_care) %>%
      select(myvars)
    
    newly_lost_r <-  bind_rows(newly_lost) %>%
      select(myvars)
    
    newly_reengage_r <- bind_rows(newly_reengage) %>%
      select(myvars)
    
    in_care_r <-  bind_rows(in_care)%>%
      select(myvars)
    
    out_care_r <-  bind_rows(out_care) %>%
      select(myvars)
    
    # Take a 10% sample
    sample <- art2009 %>%
      select(patid) %>%
      bind_rows(newart %>% filter(h1yy < 2016) %>% select(patid)) %>%
      sample_frac(0.1) 
    
    sample2 <- dead_out_care_r %>% semi_join(sample, by = "patid") %>%
      bind_rows(dead_care_r %>% semi_join(sample, by = "patid")) %>%
      bind_rows(newly_lost_r %>% semi_join(sample, by = "patid")) %>%
      bind_rows(newly_reengage_r %>% semi_join(sample, by = "patid")) %>%
      bind_rows(in_care_r %>% semi_join(sample, by = "patid")) %>%
      bind_rows(out_care_r %>% semi_join(sample, by = "patid"))
    
    sample2 <- sample2 %>%
      arrange(patid, calyy)
    return(sample2)
  } else {
    
    ############################################
    # Prepare final datasets to output
    ############################################
    # Reduce all datasets  
    dead_out_care_r <- bind_rows(dead_out_care)
    dead_out_care_r <- dead_out_care_r %>% mutate(status = 1)
    
    dead_care_r <- bind_rows(dead_care) 
    
    newly_lost_r <- bind_rows(newly_lost)
    
    newly_reengage_r <- bind_rows(newly_reengage)
    
    in_care_r <- bind_rows(in_care)
    
    out_care_r <- bind_rows(out_care)
    
    # How many times did someone leave
    n_times_lost <- newly_lost_r %>%
      group_by(patid) %>%
      tally %>%
      ungroup %>%
      rename(n_times = n) %>%
      group_by(n_times) %>%
      tally %>%
      ungroup %>%
      mutate(pct = n/sum(n)*100)
    
    # 1st incidence of being out of care: 2010-2015
    ## All alive at the beginning of 2010 + new H1YYs 2010-2015 = denominator 
    denom <- in_care_r %>% 
      filter(calyy==2009) %>% 
      select(patid, group, sex) %>%
      bind_rows(newart %>% filter(h1yy >= 2010, h1yy <= 2015) %>% select(patid, group, sex))
    
    ## All who had a newly_lost incidence between 2010-2015 and appeared in the denominator dataset
    numer <- newly_lost_r %>%
      semi_join(denom, by = "patid") %>%
      filter(ltfu_year <= 2015) %>%
      select(patid, group, sex) %>%
      distinct
    
    prop_ltfu <- numer %>% 
      group_by(group, sex) %>% 
      tally %>% 
      ungroup %>% 
      left_join(denom %>% group_by(group, sex) %>% tally %>% ungroup %>% rename(N = n), by = c("group", "sex")) %>%
      mutate(pct = n/N*100) %>%
      select(group, sex, pct)
    
    # Proportion of those out of care who died
    a <- dead_out_care_r %>% group_by(group, sex, calyy, agecat, status) %>% tally
    b <- dead_care_r %>% group_by(group, sex, calyy, agecat, status) %>% tally
    c <- newly_lost_r %>% group_by(group, sex, calyy, agecat, status) %>% tally
    d <- newly_reengage_r %>% group_by(group, sex, calyy, agecat, status) %>% tally
    e <- in_care_r %>% group_by(group, sex, calyy, agecat, status) %>% tally
    f <- out_care_r %>% group_by(group, sex, calyy, agecat, status) %>% tally
    
    # single year age tally'ing
    aa <- dead_out_care_r %>% group_by(group, sex, calyy, age) %>% tally %>% ungroup
    bb <- dead_care_r %>% group_by(group, sex, calyy, age) %>% tally %>% ungroup
    cc <- newly_lost_r %>% group_by(group, sex, calyy, age) %>% tally %>% ungroup
    dd <- newly_reengage_r %>% group_by(group, sex, calyy, age) %>% tally %>% ungroup
    ee <- in_care_r %>% group_by(group, sex, calyy, age) %>% tally %>% ungroup
    ff <- out_care_r %>% group_by(group, sex, calyy, age) %>% tally %>% ungroup
    
    # Age dist of those in care
    pcntdat <- in_care_r %>%
      group_by(group, sex, calyy, agecat) %>%
      tally %>%
      ungroup %>%
      spread(agecat, n)
    colnames(pcntdat)[grepl("2|3|4|5|6|7", colnames(pcntdat))] <- paste0("age", colnames(pcntdat)[grepl("2|3|4|5|6|7", colnames(pcntdat))])
    
    pcntdat2 <- pcntdat %>%
      gather(agecat, n, age2:age7)
    
    pcntdat2$agecat <- factor(pcntdat2$agecat,
                              levels = c("age2", "age3", "age4", "age5", "age6", "age7"),
                              labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"))
    
    pcntdat2$agecat <- factor(pcntdat2$agecat, levels=rev(levels(pcntdat2$agecat)))
    
    # Time out of care
    time_out <- newly_reengage_r %>%
      mutate(year_out = calyy - ltfu_year) %>%
      group_by(group, sex, year_out) %>%
      tally
    
    # # Unique patients who were out of care 2010-2015
    any_out <- dead_out_care_r %>% filter(calyy >= 2010, calyy <= 2015) %>% select(patid, group, sex) %>%
      bind_rows(out_care_r %>% filter(calyy >= 2010, calyy <= 2015) %>% select(patid, group, sex)) %>%
      bind_rows(newly_reengage_r %>% filter(calyy >= 2010, calyy <= 2015) %>% select(patid, group, sex)) %>%
      select(patid, group, sex) %>%
      distinct %>%
      group_by(group, sex) %>%
      tally %>%
      ungroup
    
    # Create final output list
    output_list <- list(n_times_lost = n_times_lost, 
                        dead_out_care = a, 
                        dead_care = b,
                        newly_lost = c,
                        newly_reengage = d,
                        in_care = e,
                        out_care = f,
                        #transed = pcntdat2, 
                        ini = newart_n,
                        time_out = time_out,
                        prop_ltfu = prop_ltfu,
                        n_out_2010_2015 = any_out,
                        newart_age = newart_age,
                        dead_out_care_age = aa,
                        dead_care_age = bb,
                        newly_lost_age = cc,
                        newly_reengage_age = dd,
                        in_care_age = ee,
                        out_care_age = ff)
    
    return(output_list)
  }
  
}


######################################################################################
# Replicate the wrapper function r times: FINAL
######################################################################################

# Read in probability to reengage and filter
groups <- read.csv(paste0(paramwd, '/prob_reengage.csv'))
groups <- filterfx(groups, filtergroup) 
print(groups)
setwd(outwd)

par_replicate <- function(group_row, nreps) { 
  gr=group_row['group']
  s =group_row['sex'] 
  outfile = paste0(outwd,'/',gr,'_',tolower(s),'.rda')     
  out <- parSapply(cl, 1:nreps, wrapper, groupname=gr, sexvalue=s, prob_reengage=group_row['prob'],  
                   linelist=0, simplify=F) 
  save(out, file=outfile) }

serial_replicate <- function(group_row, nreps) { 
  gr=group_row['group']
  s =group_row['sex'] 
  outfile = paste0(outwd,'/',gr,'_',tolower(s),'.rda')     
  out <- sapply(1:nreps, wrapper, groupname=gr, sexvalue=s, prob_reengage=group_row['prob'],  
                   linelist=0, simplify=F) 
  save(out, file=outfile) }

if (args$parallel) {
  # Decide number of cores
  if (args$cores == 0) {
    nCores <- detectCores() - 1
  } else {
    nCores <- args$cores
  }
  
  # Make the cluster and set RNG
  cl <- makeCluster(nCores, type='FORK', outfile='')
  clusterSetRNGStream(cl)

  print(system.time({
    apply(groups, 1, par_replicate, args$replications)
  }))
  } else {
  print(system.time({  
    apply(groups, 1, serial_replicate, args$replications)
  }))
}

