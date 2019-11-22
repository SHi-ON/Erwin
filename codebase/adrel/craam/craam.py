import os

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2

os.chdir('codebase/adrel/craam')

rcraam = importr('rcraam')
base = importr('base')
dplyr = importr('dplyr')
reader = importr('readr')

print(base.R_home())

rpy2.__path__
os.environ['R_HOME']
robjects.r('''

mdp <- read_csv("riverswim_mdp.csv")

sol <- solve_mdp(mdp, 0.99)

aggregation = data.frame(idstate = c(0,1,2,3,4,5), idstate_agg = c(0, 0, 1, 2, 3, 4), weights = c(0.5, 0.5, 1, 1, 1, 1))


mdp_agg <- 
  inner_join(mdp, aggregation, by = c("idstatefrom" = "idstate")) %>%
  mutate(idstatefrom = idstate_agg) %>% select(-idstate_agg) %>%
  inner_join(aggregation %>% select(-weights), by = c("idstateto" = "idstate")) %>%
  mutate(idstateto = idstate_agg) %>% select(-idstate_agg) %>%
  mutate(probability = probability * weights, reward = reward * weights) %>%
  group_by(idstatefrom, idaction, idstateto) %>%
  summarize(probability = sum(probability), reward = sum(reward))


mdp_agg %>% group_by(idstatefrom, idaction) %>% summarize(sum(probability))

sol_agg <- solve_mdp(mdp_agg, 0.99)

print(sol)
print(sol_agg)
  
''')
len(robjects.r['sol'])
robjects.r('sol$valuefunction$idstate')
robjects.r('sol$valuefunction')