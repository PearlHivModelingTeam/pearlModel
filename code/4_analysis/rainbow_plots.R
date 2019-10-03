library(tidyverse)
library(haven)
library(lubridate)
library(forcats)
library(scales)
library(RColorBrewer)
library(openxlsx)
library(R.utils)
library(feather)

datawd1 <- filePath(getwd(), '/../../data/input/')
output_dir <- filePath(getwd(), '/../../out/rainbow_plots')

all <- read_feather(filePath(output_dir, '/sim2.feather')) #%>% group_by(group) %>% nest()
pop09_15care <- read_feather(filePath(output_dir, 'naaccord_in_care.feather')) %>% 
  mutate(sex = replace(sex, sex=='Males', 'male'),
         sex = replace(sex, sex=='Females', 'female')) %>%
  unite(group, group, sex)

pop09_15care <- pop09_15care[!grepl('all', pop09_15care$group),]
#pop09_15care <- pop09_15care %>% group_by(group) %>% nest()

print(pop09_15care)

# Re-structure NA-ACCORD "in care" dataset (all cohorts)
pop09_15care <- pop09_15care %>%
  select(group, calyy, agecat, median_pct)

# Bind simulated and observed data
overall <- pop09_15care %>%
  filter(!grepl("_all", group)) %>%
  mutate(dataset = 1) %>%
  bind_rows(all %>% mutate(dataset = 2))

# Format factors
overall$dataset <- factor(overall$dataset,
                          levels = c(1,2),
                          labels = c("NA-ACCORD", "Simulated"))

overall$agecat <- factor(overall$agecat,
                         levels = c("2", "3", "4", "5", "6", "7"),
                         labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"))

overall$agecat <- factor(overall$agecat, levels=rev(levels(overall$agecat)))


# Create a cumulative sum variable that will be used in graphs for NA-ACCORD dataset ONLY
overall <- overall %>%
  arrange(dataset, group, calyy, desc(agecat)) %>%
  group_by(dataset, group, calyy) %>%
  mutate(cum_pct=cumsum(median_pct)) %>%
  ungroup

# Nest by group and sex
overall <- overall %>%
  group_by(group) %>%
  nest

print(overall)

# Define the graphing function
ggfx <- function(DF, group, size1, size2) {
  plot <- ggplot() +
    geom_area(data = DF %>% filter(dataset=="Simulated"), 
              stat="identity", 
              alpha = 0.9, 
              aes(x = calyy, y = p25_pct, color = agecat), 
              linetype = 3, 
              inherit.aes = F, 
              fill = NA) +
    geom_area(data = DF %>% filter(dataset=="Simulated"), 
              stat="identity", 
              alpha = 0.9, 
              aes(x = calyy, y = p75_pct, color = agecat), 
              linetype = 3, 
              inherit.aes = F, 
              fill = NA) +
    geom_area(data = DF %>% filter(dataset=="Simulated"), 
              stat="identity", 
              alpha = 0.9, 
              aes(x = calyy, y = median_pct, color = agecat, fill = agecat), 
              linetype = 1, 
              inherit.aes = F) +
    geom_point(data = DF %>% filter(dataset=="NA-ACCORD", calyy <= 2015), 
               aes(x = calyy, y = cum_pct, group = agecat, color = agecat), show.legend = F, color = "black", size = 3) +
    geom_vline(xintercept = 2015, linetype = 3) +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          panel.background = element_blank(), 
          axis.line = element_line(colour = "black"),
          axis.title = element_text(face="bold", size = size1), 
          axis.text = element_text(size = size2), 
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          legend.text = element_text(size = size2), 
          legend.title = element_text(size = size2),
          plot.title = element_text(size = size1)) + 
    scale_y_continuous(breaks = seq(0,100, by = 10)) +
    scale_x_continuous(breaks = c(seq(2009, 2030, 3), 2030)) +
    labs(x="Year", 
         y = "Cumulative %", 
         title=paste0("Age distribution of people in care: ", group),
         subtitle = "Points = NA-ACCORD, lines = Simulated populations",
         color = "Age group",
         fill = "Age group") + 
    guides(colour = guide_legend(nrow = 1, reverse = T),
           fill = guide_legend(nrow = 1, reverse = T)) +
    scale_color_viridis_d() +
    scale_fill_viridis_d()

  ggsave(filename=filePath(output_dir,paste0(group, '.png')), plot=plot, height=8, width=13, device='png')
  
}

overall2 <- overall %>%
  mutate(plot = pmap(list(data, group, 30, 28), ggfx),
         filename = paste0(group, "_rainbow_plot", ".png"))

