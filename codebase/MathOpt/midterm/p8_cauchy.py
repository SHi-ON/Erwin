# by Shayan Amani
# https://shayanamani.com

from scipy.optimize import minimize

from mid_util import *

z0 = np.array([0, 1])
x_k = z0

del_k = 0.5


# del_k = 2


def ps_calc():
    def obj_fun(p):
        return f(x_k) + grad_f(x_k).T @ p

    # According to a marked note
    # on page 68, Numerical Optimization, J. Nocedal
    def cons_fun(p):
        return del_k ** 2 - (p.T @ p)

    cons = {'type': 'ineq', 'fun': cons_fun}

    sol_ps = minimize(obj_fun, x_k, method='SLSQP', constraints=cons)
    sol_ps

    return sol_ps.x


def tau_calc(ps):
    def obj_fun(t):
        return f(x_k) + grad_f(x_k).T @ (t * ps) + 0.5 * (t * ps).T @ hess_f(x_k) @ (t * ps)

    def cons_fun(t):
        return del_k ** 2 - ((t * ps).T @ (t * ps))

    cons = {'type': 'ineq', 'fun': cons_fun}

    bnds = ((0, None), (0, None))
    # bnds = ((None, 0), (None, 0))

    # sol_t = minimize(obj_fun, x_k, constraints=cons, bounds=bnds, method='SLSQP')
    sol_t = minimize(obj_fun, x_k, constraints=cons, method='SLSQP')
    sol_t

    return sol_t.x


def tau1_calc(ps):
    def obj_fun(v):
        return f(x_k) + grad_f(x_k).T @ (v) + 0.5 * (v).T @ hess_f(x_k) @ (v)

    def cons_fun(v):
        return del_k ** 2 - ((v).T @ (v))

    cons = {'type': 'ineq', 'fun': cons_fun}
    sol_t = minimize(obj_fun, x_k, constraints=cons, method='SLSQP')
    sol_t

    return sol_t.x


if __name__ == "__main__":
    ps = ps_calc()
    ps

    tau = tau_calc(ps)
    tau

    pc = tau * ps
    pc

    ####### Calculations

    init = np.array([[0], [1]])

    g_k = np.array([[1], [9]])
    g_k

    cal_pc1 = - (0.5 / np.sqrt(82)) * g_k
    cal_pc1
    init + cal_pc1

    delta_k = 2
    mult = ((np.sqrt(82) ** 3) / (delta_k * (15 + 9 * 82)))

    cal_pc2 = - ((2 * (mult)) / np.sqrt(82)) * g_k
    cal_pc2
    init + cal_pc2


