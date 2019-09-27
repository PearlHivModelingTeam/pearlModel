suppressMessages(library(tidyverse))
suppressMessages(library(feather))


cwd <- getwd()
in_dir <- paste0(cwd, '/../../out/r')
out_dir <- paste0(cwd, '/../../out/feather/r')

load_r_data <- function(file, r_dir){
  setwd(r_dir)

  data <- get(load(file))
  data <- unlist(data, recursive=F)

  return(data)
}

format_count_data <- function(data, table_name_1, table_name_2){
  df <- data[names(data)==table_name_1]
  names(df) = seq(0,999)
  df <- bind_rows(df, .id='replication') %>% ungroup
  df <- df %>% 
    mutate(sex = replace(sex, sex=='Males', 'male'),
           sex = replace(sex, sex=='Females', 'female')) %>%
    unite(group, group, sex) %>%
    rename(year = calyy, age_cat = agecat) %>%
    select(group, replication, year, age_cat, n)

  age_cat <- seq(2, 7)
  df <- df %>% complete(age_cat, nesting(group, replication, year), fill = list(n = 0))

  if((table_name_1=='new_in_care') | (table_name_1=='new_out_care')){
    df <- df %>% mutate(year = (year + 1))
  }

  # Add missing data for new in and out of care
  if(!missing(table_name_2)){
    df2 <- data[names(data)==table_name_2]
    names(df2) <- seq(0,999)
    df2 <- bind_rows(df2, .id='replication') %>% ungroup
    df2 <- df2 %>%
      mutate(sex = replace(sex, sex=='Males', 'male'),
             sex = replace(sex, sex=='Females', 'female')) %>%
      unite(group, group, sex) %>%
      rename(year = calyy, age_cat = agecat) %>%
      select(group, replication, year, age_cat, n)

    df <- df %>% bind_rows(df2)
    df <- df %>% group_by(group, replication, year, age_cat) %>%
      summarise(n = sum(n))
  }
  return(df)
}

format_age_data <- function(data, table_name_1, table_name_2){
  df <- data[names(data)==table_name_1]
  names(df) = seq(0,999)
  df <-bind_rows(df, .id='replication') %>% ungroup
  df <- df %>% 
    mutate(sex = replace(sex, sex=='Males', 'male'),
           sex = replace(sex, sex=='Females', 'female')) %>%
    unite(group, group, sex) %>%
    rename(year = calyy) %>%
    select(group, replication, year, age, n)
  age <- seq(18, 85)
  df <- df %>% complete(nesting(group, replication, year), age, fill = list(n = 0))
  
  if((table_name_1=='dead_care_age') | (table_name_1=='dead_out_care_age')){
    df <- df %>% filter(age <= 85)
  }
  
  if((table_name_1=='newly_reengage_age') | (table_name_1=='newly_lost_age')){
    df <- df %>% mutate(year = (year + 1))
  }

  # Add missing data for new in and out of care
  if(!missing(table_name_2)){
    df2 <- data[names(data)==table_name_2]
    names(df2) <- seq(0,999)
    df2 <- bind_rows(df2, .id='replication') %>% ungroup
    df2 <- df2 %>%
      mutate(sex = replace(sex, sex=='Males', 'male'),
             sex = replace(sex, sex=='Females', 'female')) %>%
      unite(group, group, sex) %>%
      rename(year = calyy) %>%
      select(group, replication, year, age, n)

    df <- df %>% bind_rows(df2)
    df <- df %>% group_by(group, replication, year, age) %>%
      summarise(n = sum(n))
  }
  return(df)
}

format_init_count <- function(data){
  df <- data[names(data)=='new_init']
  names(df) = seq(0,999)
  df <- bind_rows(df, .id='replication') %>% ungroup
  df <- df %>% 
    mutate(sex = replace(sex, sex=='Males', 'male'),
           sex = replace(sex, sex=='Females', 'female')) %>%
    unite(group, group, sex) %>%
    select(group, replication, h1yy, n)
  return(df)
}

format_init_age <- function(data){
  df <- data[names(data)=='newart_age']
  names(df) = seq(0,999)
  df <- bind_rows(df, .id='replication') %>% ungroup
  df <- df %>% 
    mutate(sex = replace(sex, sex=='Males', 'male'),
           sex = replace(sex, sex=='Females', 'female')) %>%
    unite(group, group, sex) %>%
    select(group, replication, h1yy, age, n)
  return(df)
}

group_names <- c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male', 'idu_hisp_male',
                'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male', 'het_black_male', 'het_hisp_male',
                'het_white_female', 'het_black_female', 'het_hisp_female')
#group_names <- c('idu_white_female')

file_names <- paste0(group_names, 's.rda')

all_data <- lapply(file_names, load_r_data, in_dir)

in_care_count <- lapply(all_data, format_count_data, 'in_care', 'new_out_care') 
in_care_age <- lapply(all_data, format_age_data, 'in_care_age', 'newly_lost_age')

out_care_count <- lapply(all_data, format_count_data, 'out_care', 'new_in_care')
out_care_age <- lapply(all_data, format_age_data, 'out_care_age', 'newly_reengage_age')

reengaged_count <- lapply(all_data, format_count_data, 'new_in_care')
reengaged_age <- lapply(all_data, format_age_data, 'newly_reengage_age')

ltfu_count <- lapply(all_data, format_count_data, 'new_out_care')
ltfu_age <- lapply(all_data, format_age_data, 'newly_lost_age')

dead_in_care_count <- lapply(all_data, format_count_data, 'dead_in_care')
dead_in_care_age <- lapply(all_data, format_age_data, 'dead_care_age')

dead_out_care_count <- lapply(all_data, format_count_data, 'dead_out_care')
dead_out_care_age <- lapply(all_data, format_age_data, 'dead_out_care_age')

new_init_count <- lapply(all_data, format_init_count)
new_init_age <- lapply(all_data, format_init_age)

for (index in seq_along(group_names)){
  write_feather(as.data.frame(in_care_count[index]), paste0(out_dir, '/', group_names[index], '_in_care_count.feather'))
  write_feather(as.data.frame(in_care_age[index]), paste0(out_dir, '/', group_names[index], '_in_care_age.feather'))

  write_feather(as.data.frame(out_care_count[index]), paste0(out_dir, '/', group_names[index], '_out_care_count.feather'))
  write_feather(as.data.frame(out_care_age[index]), paste0(out_dir, '/', group_names[index], '_out_care_age.feather'))
  
  write_feather(as.data.frame(reengaged_count[index]), paste0(out_dir, '/', group_names[index], '_reengaged_count.feather'))
  write_feather(as.data.frame(reengaged_age[index]), paste0(out_dir, '/', group_names[index], '_reengaged_age.feather'))

  write_feather(as.data.frame(ltfu_count[index]), paste0(out_dir, '/', group_names[index], '_ltfu_count.feather'))
  write_feather(as.data.frame(ltfu_age[index]), paste0(out_dir, '/', group_names[index], '_ltfu_age.feather'))

  write_feather(as.data.frame(dead_in_care_count[index]), paste0(out_dir, '/', group_names[index], '_dead_in_care_count.feather'))
  write_feather(as.data.frame(dead_in_care_age[index]), paste0(out_dir, '/', group_names[index], '_dead_in_care_age.feather'))

  write_feather(as.data.frame(dead_out_care_count[index]), paste0(out_dir, '/', group_names[index], '_dead_out_care_count.feather'))
  write_feather(as.data.frame(dead_out_care_age[index]), paste0(out_dir, '/', group_names[index], '_dead_out_care_age.feather'))

  write_feather(as.data.frame(new_init_count[index]), paste0(out_dir, '/', group_names[index], '_new_init_count.feather'))
  write_feather(as.data.frame(new_init_age[index]), paste0(out_dir, '/', group_names[index], '_new_init_age.feather'))
}




