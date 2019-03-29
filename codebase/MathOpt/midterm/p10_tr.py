from mid_util import *


def m_k_p(x_k, p):
    return f(x_k) + grad_f(x_k).T @ p + 0.5 * p.T @ hess_f(x_k) @ p


z1 = np.array([0.1, 0.5])
x_k = z1

p = np.array([0.1, -0.5])

m_k_p(x_k, p)

num = f(x_k) - f(x_k + p)
num
den = m_k_p(x_k, np.zeros((2, 1))) - m_k_p(x_k, p)
den

num / den
