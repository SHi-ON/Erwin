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



