library(tidyverse)
suppressMessages(library(haven))
suppressMessages(library(lubridate))
suppressMessages(library(forcats))
suppressMessages(library(scales))
suppressMessages(library(RColorBrewer))
suppressMessages(library(openxlsx))
suppressMessages(library(feather))

cwd = getwd()
out_dir = paste0(cwd, '/../../out')
r_dir = paste0(out_dir, '/r')
feather_dir = paste0(out_dir, '/feather')
input_dir <- paste0(cwd, '/../../data/input')
fig_dir = paste0(out_dir, '/fig')

load_r_data <- function(file, r_dir){
  setwd(r_dir)

  data <- get(load(file))
  data <- unlist(data, recursive=F)

  return(data)
}

format_r_data <- function(data, category_str){
  print(category_str)
  df_r <- data[names(data)==category_str]

  names(df_r) = seq(0,99)
  df_r <- bind_rows(df_r, .id='replication') %>% ungroup
  if('calyy' %in% colnames(df_r)){
    df_r <- df_r %>% 
      mutate(sex = replace(sex, sex=='Males', 'male'),
             sex = replace(sex, sex=='Females', 'female')) %>%
      unite(group, group, sex) %>%
      rename(year = calyy, age_cat = agecat) %>%
      select(group, replication, year, age_cat, n)
  }
  else {
    df_r <- df_r %>% 
      mutate(sex = replace(sex, sex=='Males', 'male'),
             sex = replace(sex, sex=='Females', 'female')) %>%
      unite(group, group, sex) %>%
      select(group, replication, h1yy, n)
  }

  return(df_r)
}

summarize_count <- function(df){
  overall <- df %>%
    group_by(group, year, age_cat) %>%
    summarise(mean_n = mean(n),
              p025_n = quantile(n, probs = 0.025),
              p975_n = quantile(n, probs = 0.975)) %>%
    ungroup %>%
    group_by(group, year) %>%
    mutate(mean_N = sum(mean_n),
           p025_N = sum(p025_n),
           p975_N = sum(p975_n),
           mean_pct = mean_n/mean_N*100,
           lower_ci = p025_n/p025_N*100,
           upper_ci = p975_n/p975_N*100) %>%
    ungroup

    return(overall)
}

summarize_init_count <- function(df){
  overall <- df %>%
    group_by(group, h1yy) %>%
    summarise(mean_n = mean(n),
              p025_n = quantile(n, probs = 0.025),
              p975_n = quantile(n, probs = 0.975)) %>%
    ungroup 
    
    return(overall)
}

plot <- function(group_name, df, title_str, path_str) {
  ggplot(df, aes(x=year, y=mean_n)) +
    facet_wrap(vars(age_cat)) +
    geom_line(data = df %>% filter(group == group_name, Algorithm == 'R'), aes(color=Algorithm)) +
    geom_line(data = df %>% filter(group == group_name, Algorithm == 'Python'), aes(color=Algorithm)) +
    labs(title = title_str,
         subtitle = group_name,
         x = 'Year',
         y = 'Mean Count')

  ggsave(paste0(group_name, '.png'), path=paste0(fig_dir, path_str), height=8, width=13)
}

plot_init <- function(group_name, df, title_str, path_str) {
  ggplot(df, aes(x=h1yy, y=mean_n)) +
    geom_line(data = df %>% filter(group == group_name, Algorithm == 'R'), aes(color=Algorithm)) +
    geom_line(data = df %>% filter(group == group_name, Algorithm == 'Python'), aes(color=Algorithm)) +
    labs(title = title_str,
         subtitle = group_name,
         x = 'Year',
         y = 'Mean Count')

  ggsave(paste0(group_name, '.png'), path=paste0(fig_dir, path_str), height=8, width=13)
}

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male', 'idu_hisp_male',
                'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male', 'het_black_male', 'het_hisp_male',
                'het_white_female', 'het_black_female', 'het_hisp_female')
#group_names = c('idu_hisp_female')
file_names = paste0(group_names, 's.rda')
plot_names = c('in_care', 'out_care', 'dead_in_care', 'dead_out_care', 'new_in_care', 'new_out_care')
r_names = c('in_care', 'out_care', 'dead_care', 'dead_out_care', 'newly_reengage', 'newly_lost')


r_data <- lapply(file_names, load_r_data, r_dir)

for (index in seq_along(plot_names)){
  plot_str = plot_names[index]
  r_name = r_names[index]
  print(plot_str)
  df_r <- lapply(r_data, format_r_data, r_name)
  df_r <- bind_rows(df_r)
  
  df_py <- read_feather(paste0(feather_dir, '/', plot_str, '_count.feather'))
  df_py <- df_py %>% select(group, replication, year, age_cat, n) %>% filter(group %in% group_names)

  
  df_r <- summarize_count(df_r) %>% mutate(Algorithm = 'R')
  df_py <- summarize_count(df_py) %>% mutate(Algorithm = 'Python')

  df <- bind_rows(df_r, df_py)
  df$age_cat <- factor(df$age_cat,
    levels = c('2', '3', '4', '5', '6', '7'),
    labels = c('18-29', '30-39', '40-49', '50-59', '60-69', '70+'))


  print(group_names)
  lapply(group_names, plot, df, plot_str, paste0('/', plot_str))
}

df_r <- lapply(r_data, format_r_data, 'ini')
df_r <- bind_rows(df_r)
  
df_py <- read_feather(paste0(feather_dir, '/new_init_count.feather'))
df_py <- df_py %>% select(group, replication, h1yy, n) %>% filter(group %in% group_names)

df_r <- summarize_init_count(df_r) %>% mutate(Algorithm = 'R')
df_py <- summarize_init_count(df_py) %>% mutate(Algorithm = 'Python')

df <- bind_rows(df_r, df_py)

print(group_names)
lapply(group_names, plot_init, df, 'new_init', paste0('/','new_init'))
