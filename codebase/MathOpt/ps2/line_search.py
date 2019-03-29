import numpy as np
import numpy.linalg as la

import scipy.optimize as sopt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def f_rosenbrock(a, b):
    return 100 * (b - a ** 2) ** 2 + (1 - a) ** 2


def fd_rosenbrock(a, b):
    fd_x = 400 * (a**3 - b * a) + 2 * a - 2
    fd_y = 200 * (b - a ** 2)
    return np.array([fd_x, fd_y])


fig = plt.figure()
ax = fig.gca(projection='3d')

xmesh, ymesh = np.mgrid[-2:2:50j, -2:2:50j]
fmesh = f_rosenbrock(xmesh, ymesh)

ax.contour3D(xmesh, ymesh, fmesh, 200, cmap='magma')

ax.view_init(20, 35)
fig.show()
fig.show()


def steepest_descent():
    # From calculation, it is expected that the local minimum occurs at x=9/4

    cur_x = 6  # The algorithm starts at x=6
    gamma = 0.01  # step size multiplier
    precision = 0.00001
    previous_step_size = 1
    max_iters = 10000  # maximum number of iterations
    iters = 0  # iteration counter

    df = lambda x: 4 * x ** 3 - 9 * x ** 2

    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x
        cur_x -= gamma * df(prev_x)
        previous_step_size = abs(cur_x - prev_x)
        iters += 1

    print("The local minimum occurs at", cur_x)
    # The output for the above will be: ('The local minimum occurs at', 2.2499646074278457)


if __name__ == "__main__":
    steepest_descent()

