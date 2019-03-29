## Single state

offer = 5099.99
n = 10000

print("sell farm", offer)

v = 0\
    ;
for i in range(n):
    v = (1 / 1.02) * v + 100

print("keep farm", v)

print("keep farm (geometric)", 100.0 * 1.0 / (1.0 - 1 / 1.02))

## Two states

import numpy as np

gamma = 1 / 1.02

P = np.array([[0.8, 0.2], [0.4, 0.6]])
r = [200, -20]

v = np.linalg.solve(np.eye(2) - gamma * P, r)
print("value function", v)

## Two states with a decision at any time

# TODO: use fractions even when they are integral
# dimensions: action, state from, stateto
P = np.array([[[0.8, 0.2, 1.0], [0.4, 0.6, 1.0], [0.0, 0.0, 1.0]],
              [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

# dimensions: action, state
r = np.array([[200, -20, 0], [6300, 6300, 0]])

## Value iteration

# TODO: See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html

assert P.shape[0] == r.shape[0]
assert P.shape[1] == P.shape[2]
assert P.shape[1] == r.shape[1]

gamma = 1 / 1.02
iterations = 1000
action_names = ["keep", "sell"]

v = np.zeros(P.shape[1])

for t in range(iterations):
    qvalues = np.array([r[i, :] + gamma * P[i, :, :] @ v for i in range(P.shape[0])])
    v = np.max(qvalues, axis=0)

qvalues = np.array([r[i, :] + gamma * P[i, :, :] @ v for i in range(P.shape[0])])
policy = np.argmax(qvalues, axis=0)

print("Policy:", list(np.take(action_names, policy)))
print("Value function:", v)
