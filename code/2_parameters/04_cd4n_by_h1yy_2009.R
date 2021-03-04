# Load packages ---------------------------------------------------------------
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(lubridate))

pearl_dir <- Sys.getenv("PEARL_DIR")
input_dir <- filePath(pearl_dir, '/param/raw')
intermediate_dir <- filePath(pearl_dir, '/param/intermediate')
param_dir <- filePath(pearl_dir, '/param/param')
validation_dir <- filePath(pearl_dir, '/param/validation')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

test <- read_csv(filePath(intermediate_dir, 'naaccord_2009.csv'))

# Nest by group 
test <- test %>%
  group_by(group) %>%
  nest()

# Get Mean/SD of CD4N at H1YY - stratified by H1YY ----------------------------
get_cd4n_by_h1yy <- function(group, NAACCORD) {
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

}

fit_glm_to_cd4n_by_h1yy <- function(outdat) {
  # FIT GLM TO MEAN AND SD OF SQRTCD4N
  meandat <- glm(outdat$sqrtcd4n_mean ~ outdat$H1YY)
  stddat <- glm(outdat$sqrtcd4n_sd ~ outdat$H1YY)
  
  params <- data.frame(meanint = meandat$coefficients[1],
                       meanslp = meandat$coefficients[2],
                       stdint = stddat$coefficients[1],
                       stdslp = stddat$coefficients[2])
}

test <- test %>%
  mutate(outdat = pmap(list(group, data), get_cd4n_by_h1yy))

test1 <- unnest(test, outdat) %>% select(-'data')
write_csv(test1, filePath(validation_dir, 'cd4n_by_h1yy_2009_raw.csv'))


cd4n_by_h1yy_2009 <- test %>% 
  mutate(cd4n_by_h1yy_2009 = map(outdat, fit_glm_to_cd4n_by_h1yy)) %>%
  select(c(group, cd4n_by_h1yy_2009)) %>%
  unnest(cols = cd4n_by_h1yy_2009) %>%
  rename_all(tolower)

write_csv(cd4n_by_h1yy_2009, filePath(param_dir, 'cd4n_by_h1yy_2009.csv'))
