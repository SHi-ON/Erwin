import numpy as np


def rb(x):
    return np.array((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)


def grad_rb(x):
    a = np.zeros((2, 1))
    a[0, 0] = 2 * 100 * (x[1] - x[0] ** 2) * (-2 * x[0]) - 2 * (1. - x[0])
    a[1, 0] = 2 * 100 * (x[1] - x[0] ** 2)
    return a


def hess_rb(x):
    b = np.zeros((2, 2))
    b[0, 0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    b[1, 1] = 200
    b[0, 1] = -400 * x[0]
    b[1, 0] = -400 * x[0]
    return b

