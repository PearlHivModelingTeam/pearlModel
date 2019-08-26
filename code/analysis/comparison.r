suppressMessages(library(tidyverse))
suppressMessages(library(haven))
suppressMessages(library(lubridate))
suppressMessages(library(forcats))
suppressMessages(library(scales))
suppressMessages(library(RColorBrewer))
suppressMessages(library(openxlsx))
library(feather)

cwd = getwd()
out_dir = paste0(cwd, '/../../out')
input_dir <- paste0(cwd, '/../../data/input')
fig_dir = paste0(out_dir, '/fig')

load_and_format <- function(out_dir, file){
  setwd(out_dir)

  data <- get(load(file))
  data <- unlist(data, recursive=F)

  df_r <- data[names(data)=='in_care']
  names(df_r) = seq(0,99)
  df_r <- bind_rows(df_r, .id='replication') %>% ungroup
  df_r <- df_r %>% 
    mutate(sex = replace(sex, sex=='Males', 'male'),
           sex = replace(sex, sex=='Females', 'female')) %>%
    unite(group, group, sex) %>%
    rename(year = calyy, age_cat = agecat) %>%
    select(group, replication, year, age_cat, n)


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

in_care_count_r = bind_rows(
  load_and_format(out_dir, 'msm_white_males.rda'),
  load_and_format(out_dir, 'msm_black_males.rda'),
  load_and_format(out_dir, 'msm_hisp_males.rda'),
  load_and_format(out_dir, 'het_black_males.rda'),
  load_and_format(out_dir, 'het_hisp_males.rda'),
  load_and_format(out_dir, 'het_white_males.rda'),
  load_and_format(out_dir, 'het_black_females.rda'),
  load_and_format(out_dir, 'het_hisp_females.rda'),
  load_and_format(out_dir, 'het_white_females.rda'),
  load_and_format(out_dir, 'idu_black_males.rda'),
  load_and_format(out_dir, 'idu_hisp_males.rda'),
  load_and_format(out_dir, 'idu_white_males.rda'),
  load_and_format(out_dir, 'idu_black_females.rda'),
  load_and_format(out_dir, 'idu_hisp_females.rda'),
  load_and_format(out_dir, 'idu_white_females.rda'))

#in_care_count_r <- r_data_list
#print(in_care_count_r)

groups <- distinct(in_care_count_r, group)

# Count of those in care
setwd(out_dir)
in_care_count_py <- read_feather(paste0(out_dir, '/feather/in_care_count.feather'))
in_care_count_py <- in_care_count_py %>% select(group, replication, year, age_cat, n)

in_care_sum_r <- summarize_count(in_care_count_r) %>% mutate(Algorithm = 'R')
in_care_sum_py <- summarize_count(in_care_count_py) %>% mutate(Algorithm = 'Python')

in_care_sum <- bind_rows(in_care_sum_r, in_care_sum_py)

in_care_sum$age_cat <- factor(in_care_sum$age_cat, 
  levels = c('2', '3', '4', '5', '6', '7'),
  labels = c('18-29', '30-39', '40-49', '50-59', '60-69', '70+'))

apply(groups, 1, plot, in_care_sum, 'Number Of People In Care', '/in_care')

# Count of those out of care
out_care_count_py <- read_feather(paste0(out_dir, '/feather/out_care_count.feather'))
out_care_count_py <- out_care_count_py %>% select(group, replication, year, age_cat, n)

out_care_sum_py <- summarize_count(out_care_count_py) %>% mutate(Algorithm = 'Python')


#print(out_care_sum_py)







