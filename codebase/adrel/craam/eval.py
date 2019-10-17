import numpy as np
from math import sqrt
#import matplotlib.pyplot as

P = np.array([  [0.2, 0.3, 0.5],
                [0.1, 0.6, 0.3],
                [0.9, 0.1, 0.0] ])

r = np.array([-10, 7, 3])

gamma = 0.99

Phi = np.array([[1,0], [1,0], [0,1]])
print(Phi)
D = np.diag([0.5,0.5,1])

v = np.linalg.solve( (np.eye(3) - gamma * P) , r)

# LSTD
w = np.linalg.solve(Phi.T @ D @ Phi - gamma * Phi.T @ D @ P @ Phi, Phi.T @ D @ r)

print(Phi @ w)

