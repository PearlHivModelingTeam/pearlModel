suppressPackageStartupMessages(library(tidyverse))

# Load Functions
wd <- getwd()
indir <- paste0(wd,"/../../data/processed")
load(paste0(indir,"/sense.data"))
glimpse(test)

