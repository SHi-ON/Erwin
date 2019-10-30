import numpy as np
from pulp import *

N = 9
k = 5

alpha = 0.95

Phi = np.zeros((N, k))

c = np.ones((N, 1))

P = np.eye(N)

g = np.ones((N,1))

for i in range(N):
    g[i][0] = i
# r = LpVariable.dicts("weights", ((i) for i in range(k)), cat="continuous")

#
# for c in range(N):
#     for col in range(k):
#         sum = np.transpose(c) -
#


prob = pulp.LpProblem("ALP", pulp.LpMaximize)



assignments = [(i, j) for i in range(N) for j in range(1)]
# assignment variables
r = pulp.LpVariable.dicts('data-to-cluster assignments', assignments, lowBound=0, upBound=1, cat=pulp.LpContinuous)

r1 = np.array()
for i in range(N):
    np.append(r1, LpVariable("phi".join(str(i)) , 0, None, LpInteger))

# objective
prob += np.transpose(c) @ Phi @ r

# constraint
prob += (np.eye(N) - alpha * P) @ Phi @ r <= g
# prob += g + alpha * lpSum(P @ Phi @ r) >= Phi @ r

# solve
prob.solve()
