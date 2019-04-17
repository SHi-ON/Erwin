from mid_util import *

grad_f(np.array([1, 1]))

y0 = grad_f(np.array([0.1, 0.5])) - grad_f(np.array([0, 1]))
y0

s0 = np.array([[0.1], [-0.5]])
s0

num = y0 @ y0.T
num

den = y0.T @ s0
den
np.linalg.inv(den)
1 / den[0][0]

num / den
num / den[0][0]

b0 = np.eye(2)

b1 = b0 - ((b0 @ s0 @ s0.T @ b0) / (s0.T @ b0 @ s0)) + ((y0 @ y0.T) / (y0.T @ s0))
b1
