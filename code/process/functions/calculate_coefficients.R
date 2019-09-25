######################################################################################
#' Use to get coefficient estimates to predict mortality
#' df = name of data frame within the nested df containing the NA-ACCORD population data
######################################################################################
calculate_mortality_in_care <- function(df) {
  # Create new variables
  patient <- df %>%
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
  
  mortality_year <- as_tibble(t(mortality_year))
}
