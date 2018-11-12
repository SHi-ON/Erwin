import numpy as np

offer = 500
n = 10000

print("sell farm", offer*1.02**(n-1))

v = 0
for i in range(n):
    v = 1.02 * v + 100

print("keep farm", v)

print('**********')

offer = 500
n = 10000

print("sell farm", offer)

v = 0
for i in range(n):
    v = (1/1.02) * v + 100

print("keep farm", v)

#################################################

gamma = 1/1.02

P = np.array([[0.8, 0.2], [0.4, 0.6]])

v = np.linalg.solve(np.eye(2) - gamma * P, [200, -20])
print("value function", v)
