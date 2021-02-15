# Load packages ---------------------------------------------------------------
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(lubridate))

pearl_dir <- Sys.getenv("PEARL_DIR")
input_dir <- filePath(pearl_dir, '/param/raw')
intermediate_dir <- filePath(pearl_dir, '/param/intermediate')
param_dir <- filePath(pearl_dir, '/param/param')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

test <- read_csv(filePath(intermediate_dir, 'naaccord_2009.csv'))

# Nest by group
test <- test %>%
  group_by(group) %>%
  nest()

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
  mutate(h1yy_by_age_2009 = map(data, get_h1yy_prop)) %>%
  select(c(group, h1yy_by_age_2009)) %>%
  unnest(cols = h1yy_by_age_2009) %>%
  rename_all(tolower)

write_csv(h1yy_by_age_2009, filePath(param_dir, 'h1yy_by_age_2009.csv'))


