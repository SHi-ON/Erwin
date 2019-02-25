import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f_rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_f_rosenbrock(x):
    return np.array([2 * 100 * (x[1] - x[0] ** 2) * (-2 * x[0]) - 2 * (1. - x[0]), 2 * 100 * (x[1] - x[0] ** 2)])


def grad(x, max_int):
    miter = 1
    step = .0001 / miter
    vals = []
    objectfs = []
    while miter <= max_int:
        vals.append(x)
        objectfs.append(f_rosenbrock(x))
        temp = x - step * grad_f_rosenbrock(x)
        if np.abs(f_rosenbrock(temp) - f_rosenbrock(x)) > 0.01:
            x = temp
        else:
            break
        print(x, f_rosenbrock(x), miter)
        miter += 1
    return vals, objectfs, miter


start = [-1.2, 1]
val, objectf, iters = grad(start, 50)

x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objectf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='gradient descent method', cmap='inferno')
ax.legend()
fig.show()
fig.show()

