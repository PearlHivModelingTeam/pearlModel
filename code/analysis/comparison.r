suppressMessages(library(tidyverse))
suppressMessages(library(haven))
suppressMessages(library(lubridate))
suppressMessages(library(forcats))
suppressMessages(library(scales))
suppressMessages(library(RColorBrewer))
suppressMessages(library(openxlsx))
suppressMessages(library(feather))

cwd = getwd()
out_dir = paste0(cwd, '/../../out')
input_dir <- paste0(cwd, '/../../data/input')
fig_dir = paste0(out_dir, '/fig')

load_r_data <- function(out_dir, file){
  setwd(out_dir)

  data <- get(load(file))
  data <- unlist(data, recursive=F)

  return(data)
}

format_count <- function(data, category_str){
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
    summarise(mean_n = floor(mean(n)),
              p025_n = floor(quantile(n, probs = 0.025)),
              p975_n = floor(quantile(n, probs = 0.975))) %>%
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
    summarise(mean_n = floor(mean(n)),
              p025_n = floor(quantile(n, probs = 0.025)),
              p975_n = floor(quantile(n, probs = 0.975))) %>%
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

# Load R Data
msm_white_male = load_r_data(out_dir, 'msm_white_males.rda')

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

apply(groups, 1, plot_init, new_init_sum, 'Number of HAART Initiators', '/new_init_count')
