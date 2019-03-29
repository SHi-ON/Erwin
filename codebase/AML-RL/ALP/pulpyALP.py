import numpy as np
from pulp import *

# Group 3

# Inputs
# States
n = 5
# Actions
m = 3
# Features
k = 4 #n

# Fill P matrix

P = np.zeros((m, n, n))
g = np.zeros((m, n))

action_prob = np.array([[0.7, 0.2, 0.1],
                        [0.1, 0.8, 0.1],
                        [0.1, 0.2, 0.7]])

for a in range(m):
    P[a, 0, 0] = action_prob[a, 1]
    P[a, 0, 1] = action_prob[a, 2]
    P[a, 0, n - 1] = action_prob[a, 0]
    P[a, n - 1, 0] = action_prob[a, 2]
    P[a, n - 1, n - 1] = action_prob[a, 1]
    P[a, n - 1, n - 2] = action_prob[a, 0]

    for i in range(1, n - 1):
        P[a, i, i] = action_prob[a, 1]
        P[a, i, i - 1] = action_prob[a, 0]
        P[a, i, i + 1] = action_prob[a, 2]
        g[a, i] = 5 * (1 - 2 / n * abs(i - (n - 1) / 2))

#phi = np.eye(n)
phi = np.zeros((n,k))
for x in range(n):
    for y in range(k):
        phi[ x, y ] = np.sin( np.pi*(y+1)*(x/(n-1)) )


phi_t = np.transpose(phi)

# weight
r_t = LpVariable.dicts("r", list(range(k)), 0)

c_t = np.ones((1, n))  # vector
alpha = 1 / 1.02
I = np.eye(n)

print("phi: ", phi_t)
print("g: ", g)
print("c: ", c_t)
print()

# c_t times phi
c_phi = c_t @ phi

prob = LpProblem("ALP", LpMaximize)

# objective
# prob += lpDot(left, r_t)
prob += lpSum(c_phi[0][i] * r_t[i] for i in range(k))

# Constraint
# for each action
# for each state

for i in range(m):
    # (I -aP)* phi
    Pa = P[i]
    factor_m = (I - alpha * Pa) @ phi
    # (I - a*Pa) * phi * r <= g
    for j in range(n):
        # print(factor_m[j])
        prob += lpSum([factor_m[j][e] * r_t[e] for e in range(k)]) <= g[i][j]

prob.solve()

print("Status", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("objective=", value(prob.objective))

# phi @ r
res = phi @ np.array([[value(r_t[i])] for i in range(k)])

# policy
u = np.zeros(n)
for x in range(n):
    u[x] = np.argmin([g[i, x] + alpha * np.dot(P[i, x, :], res) for i in range(m)])

print(u)
