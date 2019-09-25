# Load packages ---------------------------------------------------------------
suppressMessages(library(lubridate))
suppressMessages(library(feather))
suppressMessages(library(haven))
suppressMessages(library(R.utils))
suppressMessages(library(tidyverse))
suppressMessages(library(feather))

input_dir <- filePath(getwd(), '/../../data/input')

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_white_female',
                'idu_black_male', 'idu_black_female', 'idu_hisp_male', 'idu_hisp_female', 'het_white_male',
                'het_white_female', 'het_black_male', 'het_black_female', 'het_hisp_male', 'het_hisp_female')

# Get population of NA-ACCORD alive in 2009 and in care ----------------------
get_2009 <- function(df, dir) {
  popu2 <- df %>%
    mutate(age2009 = 2009-yob,
           startYY = year(obs_entry),
           stopYY = year(obs_exit),
           H1YY = ifelse(year(haart1date) < 2000, 2000, year(haart1date))) %>%
    filter(startYY <= 2009, 2009 <= stopYY)
  
  # Updated 04/24/19
  carestat <- read_sas(filePath(dir, 'popu16_carestatus.sas7bdat'))
  colnames(carestat) <- tolower(colnames(carestat))
  
  carestat <- carestat %>%
    filter(year==2009, in_care==1)
  
  popu2 <- popu2 %>%
    semi_join(carestat, by = c("naid"))
}

naaccord <- read_feather(filePath(input_dir, 'naaccord.feather'))

naaccord_2009 <- get_2009(naaccord, input_dir)

write_feather(naaccord_2009, filePath(input_dir, 'naaccord_2009.feather'))
