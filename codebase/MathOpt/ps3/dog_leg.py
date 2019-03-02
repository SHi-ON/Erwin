# Dog-leg trust-region optimization method
# 2019 Shayan Amani
# https://shayanamani.com

import numpy as np
from scipy.optimize import minimize

from rosenbrock import *

N_ITER = 1000
DEL_HAT = 0.2

del_0 = np.random.random() * DEL_HAT
eta = np.random.random() * 0.25
x_k_0 = np.array([[2], [1]])

del_k = del_0
x_k = x_k_0


# objective function
def obj_fun(p):
    return rb(x_k) + grad_rb(x_k).T @ p + 0.5 * p.T @ hess_rb(x_k) @ p


# constraint function
def cons_fun(p):
    return del_k ** 2 - (p.T @ p)


if __name__ == '__main__':

    for i in range(N_ITER):
        # constraints
        cons = ({'type': 'ineq', 'fun': cons_fun})

        # SciPy optimizer - Sequential Least SQuares Programming method
        # this method satisfies our approximate solution need
        sol = minimize(obj_fun, x_k, method='SLSQP', constraints=cons)
        p_sol = sol.x.reshape(2, 1)

        rho = (rb(x_k) - rb(x_k + p_sol)) / (obj_fun(np.zeros((2, 1))) - obj_fun(p_sol))

        if rho < 0.25:
            del_k = 0.25 * del_k
        elif rho > 0.75 and p_sol.T @ p_sol == del_k:
                del_k = min(2 * del_k, del_0)
        if rho > eta:
            x_k = x_k + p_sol

    print("The minimum happens at\n", x_k)

    # sanity check
    # for i in range(10000):
    #     rnd = np.random.random()
    #     assert (rnd * 0.25) < 0.25

