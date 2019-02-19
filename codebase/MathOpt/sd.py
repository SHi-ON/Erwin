import numpy as np
import numpy.linalg as la

import scipy.optimize as sopt

import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import axes3d


def f(x):
    return 0.5*x[0]**2 + 2.5*x[1]**2


def df(x):
    return np.array([x[0], 5*x[1]])


fig = pt.figure()
ax = fig.gca(projection="3d")

xmesh, ymesh = np.mgrid[-2:2:50j,-2:2:50j]
fmesh = f(np.array([xmesh, ymesh]))
ax.plot_surface(xmesh, ymesh, fmesh)

pt.axis("equal")
pt.contour(xmesh, ymesh, fmesh)

guesses = [np.array([2, 2./5])]

for i in range(100):
    x = guesses[-1]
    s = -df(x)

    def f1d(alpha):
        return f(x + alpha*s)

    alpha_opt = sopt.golden(f1d)
    next_guess = x + alpha_opt * s
    guesses.append(next_guess)

    print(next_guess)
    pt.axis("equal")
    pt.contour(xmesh, ymesh, fmesh, 50)
    it_array = np.array(guesses)
    pt.plot(it_array.T[0], it_array.T[1], "x-")
    pt.figure()
    pt.show()