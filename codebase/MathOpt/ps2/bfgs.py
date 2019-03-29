import numpy as np
from math import *
from numpy.linalg import *


def f_rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_f_rosenbrock(x):
    return np.array([2 * 100 * (x[1] - x[0] ** 2) * (-2 * x[0]) - 2 * (1. - x[0]), 2 * 100 * (x[1] - x[0] ** 2)])


def hess_f_rosenbrock(x):
    a = np.zeros((2, 2))
    a[0, 0] = 2 + 1200 * x[0] ** 2 - 400 * x[1]
    a[1, 1] = 200
    a[0, 1] = -400 * x[0]
    a[1, 0] = -400 * x[0]
    return a


def tuner(func, g, x, s, alpha, thresh):
    if abs(alpha) < thresh:
        return 1
    return (func(x + alpha * s) - func(x)) / (alpha * np.dot(g, s))


def finder(func, grad_func, x, s, sigma=10 ** -1, beta=10, thresh=0.00001):
    alpha = 1.
    # increase alpha until it gets big enough
    while tuner(func, grad_func, x, s, alpha, thresh) >= sigma:
        alpha = alpha * 2

    # BTacking
    while tuner(func, grad_func, x, s, alpha, thresh) < sigma:
        alphap = alpha / (2.0 * (1 - tuner(func, grad_func, x, s, alpha, thresh)))
        alpha = max(1.0 / beta * alpha, alphap)
    return alpha


def bfgs(func, grad_func, init, epsi=10e-8, tol=10e-6):
    # starting point
    x = init
    xold = inf
    N = np.shape(x)[0]
    H = 1.0 * np.eye(N)
    c = 1
    g = grad_func(x)
    while norm(g) > epsi and norm(xold - x) > tol:
        s = -np.dot(H, g)
        alpha = finder(func, g, x, s)
        x = x + alpha * s
        gold = g
        g = grad_func(x)
        y = (g - gold) / alpha
        dotsy = np.dot(s, y)
        if dotsy > 0:
            z = np.dot(H, y)
            # updating H
            H += np.outer(s, s) * (np.dot(s, y) + np.dot(y, z)) / dotsy ** 2 - (np.outer(z, s) + np.outer(s, z)) / dotsy
        # Implement Counter
        c += 1

    return x, c


optimum, counter_iter = bfgs(f_rosenbrock, grad_f_rosenbrock, np.array([1.2, 1.2]))
optimum2, counter_iter2 = bfgs(f_rosenbrock, grad_f_rosenbrock, np.array([-1.2, 1]))

print("-------")

print("starting point = (1.2, 1.2)")
print("Number of iterations: " + str(counter_iter))
print("Found optimum point: ", str(optimum))

print("-------")

print("starting point = (-1.2, 1)")
print("Number of iterations: " + str(counter_iter2))
print("Found optimum point: ", str(optimum2))
