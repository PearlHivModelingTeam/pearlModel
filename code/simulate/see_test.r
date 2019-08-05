suppressPackageStartupMessages(library(tidyverse))
options(dplyr.print_max = 1e9)
# Load Functions
wd <- getwd()
indir <- paste0(wd,"/../../data/processed")
load(paste0(indir,"/sense.data"))
print(test$group)
print(test$naaccord_prop_2009[3])

