# Trust-region plotting
# 2019 Shayan Amani
# https://shayanamani.com

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d


def func_custom_rb(x):
    return np.array((1 - x[0]) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2)


def func_custom_grad_rb(x):
    a = np.zeros((2, 1))
    a[0, 0] = 2 * 10 * (x[1] - x[0] ** 2) * (-2 * x[0]) - 2 * (1. - x[0])
    a[1, 0] = 2 * 10 * (x[1] - x[0] ** 2)
    return a


def func_custom_hess_rb(x):
    b = np.zeros((2, 2))
    b[0, 0] = 120 * x[0] ** 2 - 40 * x[1] + 2
    b[1, 1] = 20
    b[0, 1] = -40 * x[0]
    b[1, 0] = -40 * x[0]
    return b


def model_func(p, x_k):
    row = p.shape[1]
    col = p.shape[2]

    res = np.zeros((row, col))
    for r in range(row):
        for c in range(col):
            p_ent = np.array([[p[0][r][c]], [p[1][r][c]]])
            res[r][c] = func_custom_rb(x_k) + func_custom_grad_rb(x_k).T @ p_ent + \
                        0.5 * p_ent.T @ func_custom_hess_rb(x_k) @ p_ent
    return res


# objective function
def obj_fun(p):
    return func_custom_rb(x_k) + func_custom_grad_rb(x_k).T @ p + 0.5 * p.T @ func_custom_hess_rb(x_k) @ p


if __name__ == "__main__":
    # x_k = np.array([0, -1])
    x_k = np.array([0, 0.5])

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    x_mesh, y_mesh = np.meshgrid(x, y)
    xy_mesh = np.array([x_mesh, y_mesh])
    z_mesh = model_func(xy_mesh, x_k)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(x_mesh, y_mesh, z_mesh, 200, cmap='magma')
    ax.view_init(60, 35)
    # ax.set_title("Eq. 4.2 at x = (0, -1)")
    ax.set_title("Eq. 4.2 at x = (0, 0.5)")

    # PyCharm bug with science-view which needs this call twice!
    fig.show()
    fig.show()

    ''' Trust-region radius variation'''
    delta_k = np.linspace(0, 2, 30)
    y_delta = np.zeros((30, 2))
    for i in range(30):
        # constraints
        cons = ({'type': 'ineq', 'fun': lambda p: delta_k[i] ** 2 - (p.T @ p)})

        y_delta[i] = minimize(obj_fun, x_k, method='SLSQP', constraints=cons).x

    x_axis = np.array([i[0] for i in y_delta])
    y_axis = np.array([i[1] for i in y_delta])

    cm = plt.cm.get_cmap('viridis')
    cc = np.linspace(0, 2, 30)
    sc = plt.scatter(x_axis, y_axis, c=cc, vmin=0, vmax=2, s=30, cmap=cm)
    plt.colorbar(sc)
    # plt.title("Solutions of eq. 4.3 with variation of delta\n x_k = (0, -1)")
    plt.title("Solutions of eq. 4.3 with variation of delta\n x_k = (0, 0.5)")
    plt.show()
