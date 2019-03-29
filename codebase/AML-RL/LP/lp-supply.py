# linear programming in Python using PuLP framework
# by Shayan A.
# Sep 2018

from pulp import *

# Power supply problem to minimize cost for supplying the power from
# solar or wind power units.
# solar: 2 units cost, 1 units power
# wind: 3 units cost, 2 units power
prob = pulp.LpProblem("GreenPowerSupply", pulp.LpMinimize)

# 1) Decision Variables
x1 = LpVariable("Solar", 0, None, LpInteger)
x2 = LpVariable("Wind", 0, None, LpInteger)

# 2) Objectives
prob += 2 * x1 + 3 * x2, "Total Cost"

# 3) Constraints
prob += x1 + 2 * x2 >= 4, "Minimum Power supplied"

prob.solve()

print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("\nOptimal Solution: ", value(prob.objective))
print("Solution time: {0} sec".format(round(prob.solutionTime, 4)))
