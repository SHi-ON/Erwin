import numpy as np


def f(x):
    return 0.5 * (4 * x[0] ** 2 + x[1]) ** 2 + 4 * x[1] ** 2 - x[0] ** 2 + x[0] * x[1]


def grad_f(x):
    res = np.zeros((2, 1))
    res[0, 0] = 32 * x[0] ** 3 + 8 * x[0] * x[1] - 2 * x[0] + x[1]
    res[1, 0] = 4 * x[0] ** 2 + 9 * x[1] + x[0]
    return res


def hess_f(x):
    res = np.zeros((2, 2))
    res[0, 0] = 96 * x[0] ** 2 + 8 * x[1] - 2
    res[0, 1] = 8 * x[1] + 1
    res[1, 0] = 8 * x[0] + 1
    res[1, 1] = 9
    return res
