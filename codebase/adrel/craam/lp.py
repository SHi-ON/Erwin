
import pandas as pd
import numpy as np

# import Gurobi but don't crash if it wasn't loaded
import warnings

try:
	import gurobipy as G
except ImportError:
	warnings.warn("Gurobi is required!")


mdp = pd.read_csv("codebase/adrel/craam/riverswim_mdp.csv")


lp = G.Model()

# V >= r + disc * P * V
# (disc * P - I) V = r

lp.setObjective()

for s,v in state_vars.items():
	lp.addConstr(v >= mdp.terminal_reward(s))

lp.update()
lp.optimize()


