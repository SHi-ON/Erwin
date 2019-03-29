# -*- coding: utf-8 -*-
"""
@author: Nick and Shayan
"""
import numpy as np
import pandas as pd

## Two states with a decision at any time

# TODO: use fractions even when they are integral
# dimensions: action, state from, stateto


data_ = pd.read_csv('inventory_100_100_0.csv')

data_size = data_.shape

num_states = 1 + int(pd.to_numeric(data_.loc[data_['idstateto'].idxmax()].loc['idstateto']))
num_actions = 1 + int(pd.to_numeric(data_.loc[data_['idaction'].idxmax()].loc['idaction']))

P = np.zeros((num_actions, num_states, num_states))
r_j = np.zeros((num_actions, num_states, num_states))
r = np.zeros((num_actions, num_states))

for J in range(data_size[0]):
    a = pd.to_numeric(data_.loc[J, 'idaction'])
    i = pd.to_numeric(data_.loc[J, 'idstatefrom'])
    j = pd.to_numeric(data_.loc[J, 'idstateto'])

    # dimensions: action, state from, stateto
    P[a, i, j] = pd.to_numeric(data_.loc[J, 'probability'])
    r_j[a, i, j] = pd.to_numeric(data_.loc[J, 'reward'])

# Reward depending on end-state
for s in range(num_states):
    for a in range(num_actions):
        r[a, s] = np.sum(P[a, s, :] * r_j[a, s, :])
#
# P = np.array([[[0.8, 0.2, 1.0],
#               [0.4, 0.6, 1.0], 
#               [0.0, 0.0, 1.0] ],
#              
#              [[0.0, 0.0, 1.0],
#               [0.0, 0.0, 1.0], 
#               [0.0, 0.0, 1.0] ]])
# r =np.array([ [200,-20,0], [6300,6300,0] ])

## Modified POLICY iteration

assert P.shape[0] == r.shape[0]
assert P.shape[1] == P.shape[2]
assert P.shape[1] == r.shape[1]

gamma = 1 / 1.02
max_iterations = 10000

# The m_n sequence
m_n = 50

# Tolerence
epsilon = 10 ** -6

# Initialize v_o the value function
v = np.zeros(P.shape[2])
p_d = np.zeros((P.shape[1], P.shape[2]))
r_d = np.zeros(P.shape[1])
d_o = np.zeros(P.shape[2])  # Decision maps state into action

for n in range(max_iterations):

    # Set new decision according to argmax
    for s in range(P.shape[1]):
        qval = np.array([r[i, :] + gamma * P[i, :, :] @ v for i in range(P.shape[0])])
        d = np.argmax(qval, axis=0)

    # Apply decision to Transition matrix by row replacement
    for s in range(P.shape[1]):
        # print(s)
        # print(P[s])
        p_d[s, :] = P[d[s], s, :]
        r_d[s] = r[d[s], s]

    #    #is identical to previous?
    #    #else store current value for later
    #    if np.all( d_o == d ):
    #        break;
    #    else:
    #        d_o = d

    # corresponding m from the sequence {m_n}
    m = m_n

    # Init u
    uu = np.array([r[i, :] + gamma * P[i, :, :] @ v for i in range(P.shape[0])])
    u = np.max(uu, axis=0)

    # Commence Partial Policy Eval
    if (np.max(np.abs(u - v)) < epsilon * (1 - gamma) / (2 * gamma)):
        # value function is within tolerence.
        # Therefore d is truly optimal within tolerance

        break;

    else:
        # Update u

        for k in range(m):
            u = r_d + gamma * p_d @ u

    v = u

# Complete loop
# ----------------------------------------------------


# Supply outputs

for s in range(r.shape[1]):
    print(s, d[s])

state = [x for x in range(d.shape[0])]

df_last = pd.DataFrame({'idstate': state, 'idaction': d}, columns=['idstate', 'idaction'])
df_last.to_csv('MPI_output.csv', index=0)
