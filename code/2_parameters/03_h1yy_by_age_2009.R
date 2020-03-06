# Load packages ---------------------------------------------------------------
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

test <- read_feather(filePath(input_dir, 'naaccord_2009.feather'))

# Nest by group 
test <- test %>%
  group_by(group) %>%
  nest(.key = "naaccord_2009")


# Get proportion of H1YY within each age group in NA-ACCORD population -------
get_h1yy_prop <- function(DF) {
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

h1yy_by_age_2009 <- test %>% 
  mutate(h1yy_by_age_2009 = map(naaccord_2009, get_h1yy_prop)) %>%
  select(c(group, h1yy_by_age_2009)) %>%
  unnest() %>% 
  rename_all(tolower)

write_feather(h1yy_by_age_2009, filePath(param_dir, 'h1yy_by_age_2009.feather'))


