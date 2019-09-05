suppressMessages(library(tidyverse))
suppressMessages(library(glue))
suppressMessages(library(feather))

# I'm color blind lol
cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


cwd <- getwd()
out_dir <- paste0(cwd, '/../../out')
r_dir <- paste0(out_dir, '/feather/r')
py_new_dir <- paste0(out_dir, '/feather/py_new')
py_old_dir <- paste0(out_dir, '/feather/py_old')
fig_dir <- paste0(out_dir, '/fig')

summarize_count <- function(df){
  df <- df %>%
    group_by(group, year, age_cat) %>%
    summarise(mean_n = mean(n),
              p025_n = quantile(n, probs = 0.025),
              p975_n = quantile(n, probs = 0.975)) %>%
    ungroup
  return(df)
}

plot <- function(group_name, df, title_str, path_str) {
  ggplot(df, aes(x=year, y=mean_n)) +
    facet_wrap(vars(age_cat)) +
    geom_line(data= df %>% filter(algorithm == 'R'), aes(color=algorithm)) +
    geom_ribbon(data= df %>% filter(algorithm =='R'), aes(ymin=p025_n,ymax=p975_n),alpha=0.3) +
    geom_line(data= df %>% filter(algorithm == 'Python, no cd4 reset'), aes(color=algorithm)) +
    geom_line(data= df %>% filter(algorithm == 'Python, cd4 reset'), aes(color=algorithm)) +
    labs(title = title_str,
         subtitle = group_name,
         x = 'Year',
         y = 'Mean Count') +
    scale_colour_manual(values=cbPalette)      

  ggsave(paste0(group_name, '.png'), path=paste0(fig_dir, path_str), height=8, width=13)
}

group_names = c('msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male', 'idu_hisp_male',
                'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male', 'het_black_male', 'het_hisp_male',
                'het_white_female', 'het_black_female', 'het_hisp_female')

plot_names = c('in_care_count', 'out_care_count', 'dead_in_care_count', 'dead_out_care_count', 'new_in_care_count', 'new_out_care_count')

for(group_name in group_names){
  print(group_name)
  for(plot_name in plot_names){
    print(plot_name)
    df_r <- summarize_count(read_feather(glue('{r_dir}/{group_name}_{plot_name}.feather'))) %>% mutate(algorithm = 'R')
    df_py_new <- summarize_count(read_feather(glue('{py_new_dir}/{group_name}_{plot_name}.feather'))) %>% mutate(algorithm = 'Python, cd4 reset')
    df_py_old <- summarize_count(read_feather(glue('{py_old_dir}/{group_name}_{plot_name}.feather'))) %>% mutate(algorithm = 'Python, no cd4 reset')

    df <- bind_rows(df_r, df_py_new, df_py_old)
    df$age_cat <-factor(df$age_cat,
      levels = c('2', '3', '4', '5', '6', '7'),
      labels = c('18-29', '30-39', '40-49', '50-59', '60-69', '70+'))

    plot(group_name, df, plot_name, glue('/{plot_name}'))
  }
}

