#'Changes 09/11/18 - use 2014 and 2015 in addition to 2009-2013 to fit mixture model
#' 09/25/18: Inflate the number of new HAART initiators by 29% (black MSM), 38% (Hisp MSM), and 37% (white MSM)
#' 09/26/18: Base the mortality function on 2009-2015 not 2005-2015
#' 10/18/18: Update inflation factors
#' 01/25/19: Drop the inflation factors because we are now accounting for loss to follow-up in the model
#' 05/30/19: Incorporate prediction intervals

######################################################################################
#' Use to filter the dataset to the risk/race groups of interest
#' DF = name of data frame
#' VALUE = value(s) of group variable to filter on
######################################################################################
filterfx <- function(DF, VALUE) {
  filtered <- DF %>%
    filter(group %in% VALUE)
}


clean_coeff <- function(df) {
  colnames(df) <- tolower(colnames(df))
  df$pop2 <- tolower(df$pop2)
  df$sex[df$sex==1] <- "Males"
  df$sex[df$sex==2] <- "Females"
  df <- df %>% rename(group = pop2)
  
  return(df)
}

####################################################################################
# Define percentiles from time from haart to model cd4 increase
####################################################################################
p5 <- 1
p35 <- 3
p65 <- 6
p95 <- 12

######################################################################################
#' Use to get coefficient estimates to predict mortality
#' DF = name of data frame within the nested DF containing the NA-ACCORD population data
######################################################################################
mortality_fx <- function(DF) {
  # Create new variables
  patient <- DF %>%
    mutate(h1yy = year(haart1date),
           sqrtcd4n = ifelse(cd4n >=0, sqrt(cd4n), NA),
           deathyy = year(deathdate),
           entryyy = year(obs_entry),
           exityy = year(obs_exit))
  
  # Convert from wide to long - 1 row / year of study observation for each patient
  yearly <- patient %>%
    nest(entryyy, exityy) %>%
    mutate(year = map(data, ~seq(unique(.x$entryyy), unique(.x$exityy), 1))) %>%
    unnest(year)
  
  # Recode variables
  yearly <- yearly %>%
    mutate(age = year - yob,
           agecat = floor(age/10), 
           agecat = replace(agecat, agecat < 2, 2),
           agecat = replace(agecat, agecat > 6, 6),
           ageby10 = floor(age/10),
           ageby10 = replace(ageby10, ageby10==1, 2),
           ageby10 = replace(ageby10, ageby10 > 7, 7),
           realdeath = 0,
           realdeath = replace(realdeath, year==deathyy, 1),
           py = ifelse(year == year(obs_entry) & year == year(obs_exit), (obs_exit - obs_entry + 1) / 365.25,
                       ifelse(year == year(obs_entry), (make_date(year = year, month = 12, day = 31) - obs_entry + 1) / 365.25,
                              ifelse(year==year(obs_exit), (obs_exit - make_date(year = year, mont = 1, day = 1) + 1) / 365.25, 1))),
           logpy = log(py)) %>%
    filter(year >= 2009, # modified 9/26/18
           year <= 2015) # added 09/11/18
  
  # Drop patients with any missing predictor variable that will be used in regression
  yearly2 <- yearly %>%
    select(naid, realdeath, ageby10, year, h1yy, sqrtcd4n) %>%  
    na.omit() 
  
  # Run regression model
  mylogit <- geeglm(realdeath ~ ageby10 + sqrtcd4n + year + h1yy, 
                    id = naid, 
                    data = yearly2, 
                    corstr = "unstructured", 
                    family=binomial(link='logit'))
  
  # Extract model coefficients into a data frame
  mortality_year <- mylogit$coefficients
  
  names(mortality_year) <- c("intercept_est", "ageby10_est", "sqrtcd4n_est", "year_est", "h1yy_est")
  
  mortality_year <- data.frame(t(mortality_year))
}

######################################################################################
#' 2009 population of persons in the US living with diagnosed HIV and on HAART -
#'      TIME-FIXED COMPONENTS
######################################################################################
# 1. Get population of NA-ACCORD particiapnts alive in 2009 and (as of 04/24/19) in care
fx1 <- function(DF, dir) {
  popu2 <- DF %>%
    mutate(age2009 = 2009-yob,
           startYY = year(obs_entry),
           stopYY = year(obs_exit),
           H1YY = ifelse(year(haart1date) < 2000, 2000, year(haart1date))) %>%
    filter(startYY <= 2009, 2009 <= stopYY)
  
  # Updated 04/24/19
  carestat <- read_sas(paste0(dir,"/popu16_carestatus.sas7bdat"))
  colnames(carestat) <- tolower(colnames(carestat))
  
  carestat <- carestat %>%
    filter(year==2009, in_care==1)
  
  popu2 <- popu2 %>%
    semi_join(carestat, by = c("naid"))
}

# 2. Fit a weibull distribution to age in 2009
fx2 <- function(DF) {
  f1 <- fitdistr(DF$age2009, "weibull", lower=c(0,0))
  
  shapepar <- f1$estimate[1]
  scalepar <- f1$estimate[2]
  
  param <- data.frame(shape = shapepar, scale = scalepar)
}

# 3. Fit a mixed normal distribution to age in 2009 (**2x more iterations for IDUs vs MSM**)
fx3 <- function(DF) {
  
  f2 <- normalmixEM(DF$age2009, k = 2, mu = c(30,50), sigma = c(1,1), lambda = c(0.5,0.5), maxit = 4000)
  
  mus <- c(f2$mu)
  sigmas <- c(f2$sigma)
  lambdas <- c(f2$lambda)
  
  params <- data.frame(mu1 = mus[1],
                       mu2 = mus[2],
                       sigma1 = sigmas[1],
                       sigma2 = sigmas[2],
                       lambda1 = lambdas[1],
                       lambda2 = lambdas[2])
}

# 4. Get proportion of H1YY within each age group in NA-ACCORD population
fx4 <- function(DF) {
  # Define age groups
  popu2 <- DF %>%
    mutate(age2009cat = floor(age2009/10)) %>%
    mutate(age2009cat = replace(age2009cat, age2009cat > 7, 7)) 
  
  # Get proportion of HAART initiation year within each age group
  tabdat <- popu2 %>%
    arrange(age2009cat, H1YY) %>%
    group_by(age2009cat, H1YY) %>%
    tally %>%
    mutate(pct=n/sum(n)) %>%
    ungroup %>%
    select(age2009cat, H1YY, pct)
  
  # Expand to include missing H1YY, n, and pct values for each age2009cat
  tabdat <- tabdat %>%
    full_join(popu2 %>% expand(age2009cat, H1YY), by=c("age2009cat", "H1YY")) %>%
    mutate(pct = replace(pct, is.na(pct), 0)) %>%
    arrange(age2009cat, H1YY)
  
  # Define lower and upper limits of cumulative distributions
  caldist <- tabdat %>%
    group_by(age2009cat) %>%
    mutate(upper = cumsum(pct),
           lower = coalesce(lag(upper),0)) %>%
    ungroup
}

# 5. Get Mean/SD of CD4N at HAART initiation - stratified by H1YY
fx5 <- function(NAACCORD) {
  popu2 <- NAACCORD %>%
    mutate(startYY = year(obs_entry),
           stopYY = year(obs_exit),
           H1YY = year(haart1date),
           sqrtcd4n = ifelse(cd4n >= 0, sqrt(cd4n), NA)) %>%
    filter(startYY <= 2009, 2009 <= stopYY)
  
  # Get mean and SD sqrtcd4n by H1YY - changed from 2013 to 2009 8/24
  outdat <- popu2 %>%
    filter(H1YY >= 2000, H1YY <= 2009, sqrtcd4n >= 0) %>%
    group_by(H1YY) %>%  
    summarise(sqrtcd4n_mean = mean(sqrtcd4n),
              sqrtcd4n_sd = sd(sqrtcd4n),
              sqrtcd4n_n = n()) %>% 
    ungroup
  
  # FIT GLM TO MEAN AND SD OF SQRTCD4N
  meandat <- glm(outdat$sqrtcd4n_mean ~ outdat$H1YY)
  stddat <- glm(outdat$sqrtcd4n_sd ~ outdat$H1YY)
  
  params <- data.frame(meanint = meandat$coefficients[1],
                       meanslp = meandat$coefficients[2],
                       stdint = stddat$coefficients[1],
                       stdslp = stddat$coefficients[2])
}

######################################################################################
# 2009 - 2015 population of new INIs - TIME-FIXED COMPONENTS
######################################################################################
# 1. Fit a mixed normal distribution to age at HAART initiation for each year, 2009-2015 - ** differs for IDU vs MSM **
inifx1 <- function(NAACCORD, group, sex) {
  inipop <- NAACCORD %>%
    mutate(H1YY = year(haart1date),
           iniage = H1YY-yob) %>%
    filter(H1YY <= 2015, 2009 <= H1YY) 
  
  # New for IDU - collapse groups
  inipop <- inipop %>%
    mutate(period = case_when(group %in% c("idu_black", "idu_white") & sex=="Females" & H1YY %in% c(2009, 2010) ~ 2009,
                              group %in% c("idu_black", "idu_white")  & sex=="Females" & H1YY %in% c(2011, 2012) ~ 2011,
                              group %in% c("idu_black", "idu_white")  & sex=="Females" & H1YY %in% c(2013, 2014, 2015) ~ 2013,
                              group %in% c("idu_hisp") & sex=="Males" & H1YY %in% c(2009, 2010) ~ 2009,
                              group %in% c("idu_hisp")  & sex=="Males" & H1YY %in% c(2011, 2012) ~ 2011,
                              group %in% c("idu_hisp")  & sex=="Males" & H1YY %in% c(2013, 2014, 2015) ~ 2013,
                              group=="idu_hisp" & sex=="Females" ~ 2009,
                              TRUE ~ H1YY))
  
  inipop2 <- inipop %>%
    arrange(period) %>%
    group_by(period) %>%
    nest %>%
    mutate(mixture_model = map(data, ~(normalmixEM(maxit = 4000, .$iniage, verb=F, k = 2, mu = c(25, 50), sigma = c(1,1), lambda = c(0.5, 0.5)))),
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

# 2. Fit a GLM to each parameter from the MIXTURE model and extract predicted values of mu, sigma, and lambda parameters
# Modified for IDU
inifx2 <- function(DATA) {
  
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
    unnest
  
  colnames(DF2)[colnames(DF2)=="period"] <- "H1YY"
  
  return(DF2)
}

######################################################################################
#' Make predictions from regression models for HAART years 2009-2030
######################################################################################
predict_it <- function(model) {
  data.frame(H1YY = seq(2009,2030), pred = predict(model, type="response", newdata = data.frame(H1YY = seq(2009,2030))))
}

######################################################################################
#' 2009 population of persons in the US living with diagnosed HIV and on HAART -
#'      TIME-VARYING COMPONENTS
#' MODIFIED 10/24/18 - we know we will use mixture, so drop the weibull tests
######################################################################################
fx6 <- function(age2009_ci, on_art, naaccord_prop_2009, naaccord_cd4_2009) {
  
  stop()
  # Sample from the 95% CIs
  chk <- age2009_ci %>%
    mutate(mu1 = runif(1, min = mu1_p025, max = mu1_p975),
           mu2 = runif(1, min = mu2_p025, max = mu2_p975),
           sigma1 = runif(1, min = sigma1_p025, max = sigma1_p975),
           sigma2 = runif(1, min = sigma2_p025, max = sigma2_p975),
           lambda1 = runif(1, min = lambda1_p025, max = lambda1_p975),
           lambda2 = 1 - lambda1)
  
  # Simulate population with mixed distribution
  rtnorm_age <- function(n, mean, sd, a = 18, b = 85){
    qnorm(runif(n, pnorm(a, mean, sd), pnorm(b, mean, sd)), mean, sd)
  }
  
  components <- sample(1:2, prob=c(chk$lambda1, chk$lambda2), size=on_art, replace=TRUE)
  mus <- c(chk$mu1, chk$mu2)
  sds <- c(chk$sigma1, chk$sigma2)
  
  n_pop1 <- length(components[components==1])
  pop1 <- data.frame(age2009 = rtnorm_age(n_pop1, mus[1], sds[1]))
  
  n_pop2 <- length(components[components==2])
  pop2 <- data.frame(age2009 = rtnorm_age(n_pop2, mus[2], sds[2]))
  
  sim_mixed <- bind_rows(pop1, pop2)
  
  # Drop out of range age values and filter to on_art
  agedat <- sim_mixed %>%
    mutate(age2009 = floor(age2009)) %>%
    # Define age groups
    mutate(age2009cat = floor(age2009/10)) %>%
    mutate(age2009cat = replace(age2009cat, age2009cat > 7, 7)) 
  
  # Assign H1YY to match the distribution in NA-ACCORD
  agedat2 <- agedat %>%
    mutate(randnum = runif(nrow(agedat))) %>%
    left_join(naaccord_prop_2009, by="age2009cat") %>%
    filter(lower < randnum,
           randnum <= upper) %>%
    select(-randnum, -pct, -upper, -lower)
  
  # Assign H1MM and H1DD
  n_pats <- nrow(agedat2)
  
  agedat2 <- agedat2 %>% 
    mutate(h1mm = sample(seq(1, 12), n_pats, replace = T),
           h1dd = sample(seq(1, 28), n_pats, replace = T))
  
  # FINAL SIMULATED DATASET WITH AGE
  age2009 <- agedat2 %>%
    mutate(h1date = make_date(H1YY, h1mm, h1dd)) %>%
    select(age = age2009, agecat = age2009cat, h1date, H1YY)
  
  # ADD IN CD4N @ HAART initiation
  meanint <- naaccord_cd4_2009$meanint
  meanslp <- naaccord_cd4_2009$meanslp
  stdint <- naaccord_cd4_2009$stdint
  stdslp <- naaccord_cd4_2009$stdslp
  
  # ASSIGN CD4N (normal distribution truncated at 0)
  rtnorm <- function(n, mean, sd, a = 0, b = Inf){
    qnorm(runif(n, pnorm(a, mean, sd), pnorm(b, mean, sd)), mean, sd)
  }
  
  # REVISED 10/24/18: DROP ROWWISE CALCULATION OF SQRTCD4N B/C IT'S TIME CONSUMING
  #agetcell2009 <- age2009 %>%
  #  mutate(mn = meanint + (H1YY*meanslp),
  #         std = stdint + (H1YY*stdslp)) %>%
  #  rowwise() %>%
  #  mutate(sqrtcd4n = rtnorm(1, mn, std)) %>%
  #  ungroup
  agetcell2009 <- age2009 %>%
    mutate(mn = meanint + (H1YY*meanslp),
           std = stdint + (H1YY*stdslp)) %>%
    group_by(H1YY) %>%
    nest
  
  assign_cd4 <- function(DF) {
    out <- DF %>%
      mutate(sqrtcd4n = rtnorm(n(), mn, std))
  }
  
  agetcell2009 %>%
    mutate(sqrtcd4n = map(data, assign_cd4)) %>%
    select(H1YY, sqrtcd4n) %>%
    unnest
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



surv_fx3 <- function(DF, SURV) {
  
  # Randommly assign n_dx by year, sex, and group using TRIANGLE distribution - REVISED 8/24 TO USE A UNIFORM DISTRIBUTION
  
  fit_all3 <- DF %>%
    group_by(year) %>%
    mutate(n_dx = runif(1, lower, upper)) %>%
    select(year, n_dx) %>%
    ungroup
  
  fit_all4 <- fit_all3 %>%
    filter(year > 2015) %>%
    bind_rows(SURV) %>%
    arrange(year)
  
  fit_all4$n_dx <- floor(fit_all4$n_dx)
  
  #################################################
  # Get Number of New Initiators
  #################################################
  ##########################################
  # Estimate #s linked to care
  # We assume 75% link in the first year and 
  # 85% (equal to 40% of the remaining individuals) 
  # link within the next 3 years (year 1 to 4) 
  # with equal distibution over each year								
  ##########################################
  ini <- fit_all4 %>%
    arrange(year) %>%
    mutate(lag0 = n_dx*0.75,
           N1 = (1-0.75)*0.4*0.33*n_dx) %>%
    ungroup
  
  ini <- ini %>%
    mutate(lag1 = lag(N1),
           lag2 = lag(N1,2),
           lag3 = lag(N1, 3)) %>%
    ungroup %>%
    select(-N1)
  
  ini$total_linked <- rowSums(ini[,c("lag0", "lag1", "lag2", "lag3")], na.rm=TRUE)
  
  ini$total_haart <- floor(ini$total_linked * 0.75)
  
  # Check for negative #s
  ini <- ini %>%
    mutate(total_haart = replace(total_haart, total_haart < 1, 1))
  
  # Create final dataset of initiators
  ini <- ini %>%
    select(year, total_haart) %>%
    spread(year, total_haart)
  
  return(ini)
  
}

######################################################################################
#' 2009 - 2015 population of new INIs - TIME-VARYING COMPONENTS
#' REVISED 10/24/18 TO DROP ROWWISE CALCULATION OF NEW PREDICTED VALUES
######################################################################################
# Clean the mixed normal parameter estimates
inifx3 <- function(DATA) {
  pred18 <- DATA %>% 
    filter(H1YY==2018) %>%
    select(param, pred18 = pred)
  
  DF2 <- DATA %>%
    mutate(pred = replace(pred, pred < 0, 0))
  
  #DF2 <- DF2 %>%
  #  left_join(pred18, by="param") %>%
  #  rowwise() %>%
  #  mutate(new_pred = ifelse(H1YY <= 2018, pred, runif(1, min(pred, pred18), max(pred, pred18)))) %>%
  #  ungroup %>%
  #  select(param, H1YY, new_pred)
  
  DF2 <- DF2 %>%
    left_join(pred18, by="param") %>%
    group_by(param, H1YY) %>%
    nest
  
  predfx <- function(DF, H1YY) {
    DF2 <- DF %>%
      mutate(new_pred = ifelse(H1YY <= 2018, pred, runif(n(), min(pred, pred18), max(pred, pred18))))
  }
  
  DF2 <- DF2 %>%
    mutate(new_pred = map2(data, H1YY, predfx)) %>%
    select(param, H1YY, new_pred) %>%
    unnest %>%
    select(param, H1YY, new_pred)
  
  DF2 <- DF2 %>% 
    spread(param, new_pred) %>%
    mutate(lambda1 = replace(lambda1, !is.na(lambda1), 1 - lambda2),
           lambda1 = replace(lambda1, lambda1 < 0,0),
           lambda2 = replace(lambda2, lambda2 < 0,0),
           lambda1 = replace(lambda1, lambda1 > 1, 1),
           lambda2 = replace(lambda2, lambda2 > 1, 1)) %>%
    mutate(model_type = ifelse(lambda1==0,2,
                               ifelse(lambda2==0,1,0)))
}

# 6. Simulate the population (mixed normal)
inifx4 <- function(INI, DATA) {
  # Determine size of population to simulate
  n_ini <- data.frame(t(INI))
  n_ini <- tibble::rownames_to_column(n_ini)
  
  colnames(n_ini) <- c("H1YY", "n")
  
  n_ini$H1YY <- as.numeric(n_ini$H1YY)
  
  # Nest by h1yy and model type
  ini_age <- DATA %>%
    left_join(n_ini, by="H1YY") %>%
    arrange(H1YY) %>%
    group_by(H1YY, model_type) %>%
    nest
  
  rtnorm <- function(n, mean, sd, a = 18, b = 85){
    qnorm(runif(n, pnorm(a, mean, sd), pnorm(b, mean, sd)), mean, sd)
  }
  
  # Simulate the population
  simpop <- function(model_type, df) {
    
    # Normal (younger pop)
    if (model_type==1) {
      
      samples <- data.frame(iniage = rtnorm(df$n, df$mu1, df$sigma1))
      
    } else if (model_type==2) {
      # Normal (older pop)
      
      samples <- data.frame(iniage = rtnorm(df$n, df$mu2, df$sigma2))
      
    } else if (model_type==0) {
      # Mixed normal
      components <- sample(1:2, prob=c(df$lambda1, df$lambda2), size=df$n, replace=TRUE)
      
      # Pop 1
      n_pop1 <- length(components[components==1])
      pop1 <- data.frame(iniage = rtnorm(n_pop1, df$mu1, df$sigma1))
      
      # Pop 2
      n_pop2 <- length(components[components==2])
      pop2 <- data.frame(iniage = rtnorm(n_pop2, df$mu2, df$sigma2))
      
      samples <- bind_rows(pop1, pop2)
    }
  }
  
  ini_age <- ini_age %>%
    mutate(simpop_mixed = pmap(list(model_type, data), simpop)) %>%
    select(H1YY, simpop_mixed) %>%
    unnest
  
  # Assign HAART initiation date (assuming uniform distribution)
  n_pats <- nrow(ini_age)
  
  ini_age <- ini_age %>% 
    mutate(h1mm = sample(seq(1, 12), n_pats, replace = T),
           h1dd = sample(seq(1, 28), n_pats, replace = T)) %>%
    mutate(h1date = make_date(year = H1YY, month = h1mm, day = h1dd),
           # Added 01/28/19
           iniage = floor(iniage)) %>%
    select(iniage, h1date, H1YY)
}

# Simulate sqrtcd4n at HAART initiation among new INIs - MODIFIED FOR IDU 
ini_cd4_fx <- function(NAACCORD, GROUP, SEX, AGEINI) {
  # Go back to NA-ACCORD population & get sqrt cd4n @ HAART initiation
  popu2 <- NAACCORD %>%
    mutate(H1YY = year(haart1date),
           sqrtcd4n = ifelse(cd4n >= 0, sqrt(cd4n), NA)) %>%
    filter(2010 <= H1YY, H1YY <= 2014)
  
  # Collapse H1YYs into periods for groups with small numbers (IDU only)
  popu2 <- popu2 %>%
    mutate(period = case_when(GROUP %in% c("idu_black", "idu_white") & SEX=="Females" & H1YY %in% c(2009, 2010) ~ 2009,
                              GROUP %in% c("idu_black", "idu_white")  & SEX=="Females" & H1YY %in% c(2011, 2012) ~ 2011,
                              GROUP %in% c("idu_black", "idu_white")  & SEX=="Females" & H1YY %in% c(2013, 2014, 2015) ~ 2013,
                              GROUP %in% c("idu_hisp") & SEX=="Males" & H1YY %in% c(2009, 2010) ~ 2009,
                              GROUP %in% c("idu_hisp")  & SEX=="Males" & H1YY %in% c(2011, 2012) ~ 2011,
                              GROUP %in% c("idu_hisp")  & SEX=="Males" & H1YY %in% c(2013, 2014, 2015) ~ 2013,
                              GROUP=="idu_hisp" & SEX=="Females" ~ 2009,
                              TRUE ~ H1YY))
  
  # Get the mean sqrt CD4N by H1YY (MODIFIED FOR IDU TO USE PERIOD INSTEAD OF H1YY)
  sumdat <- popu2 %>%
    filter(!is.na(sqrtcd4n)) %>%
    group_by(period) %>%
    summarise(mean = mean(sqrtcd4n),
              sd = sd(sqrtcd4n)) %>%
    ungroup
  
  # Fit GLMs to the mean and SD (MODIFIED FOR IDU TO USE PERIOD INSTEAD OF H1YY)
  meandat <- glm(sumdat$mean ~ sumdat$period)
  stddat <- glm(sumdat$sd ~ sumdat$period)
  
  meanint <- meandat$coefficients[1]
  meanslp <- meandat$coefficients[2]
  stdint <- stddat$coefficients[1]
  stdslp <- stddat$coefficients[2]
  
  # Update 03/27/19: set slopes to 0 for IDU Hisp F
  # For Hispanic IDU Females - coalesce w/ 0
  meanslp <- coalesce(meanslp, 0)
  stdslp <- coalesce(stdslp, 0)
  
  # Assign CD4 values in a simulated population of HAART initiators, 2009-2030
  rtnorm <- function(n, mean, sd, a = 0, b = Inf){
    qnorm(runif(n, pnorm(a, mean, sd), pnorm(b, mean, sd)), mean, sd)
  }
  
  # Update 08/24 - fix the mean / std at the 2020 values
  # REVISED 10/24/18 - DROP ROWWISE CALCULATION OF SQRTCD4N
  #agetcellini <- AGEINI %>%
  #  mutate(H1YY2 = ifelse(H1YY <= 2020, H1YY, 2020)) %>%
  #  mutate(mn = meanint + (H1YY2*meanslp),
  #         std = stdint + (H1YY2*stdslp)) %>%
  #  rowwise() %>%
  #  mutate(sqrtcd4n = rtnorm(1, mn, std)) %>%
  #  ungroup
  
  agetcellini <- AGEINI %>%
    mutate(H1YY2 = ifelse(H1YY <= 2020, H1YY, 2020)) %>%
    mutate(mn = meanint + (H1YY2*meanslp),
           std = stdint + (H1YY2*stdslp)) %>%
    group_by(H1YY) %>%
    nest
  
  cd4gen <- function(DF) {
    DF <- DF %>%
      mutate(sqrtcd4n = rtnorm(n(), mn, std))
  }
  
  agetcellini <- agetcellini %>%
    mutate(sqrtcd4n = map(data, cd4gen)) %>%
    select(H1YY, sqrtcd4n) %>%
    unnest
  
}
