import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv


def f_rosenbrock(x):
    return 100 * np.square(np.square(x[0]) - x[1]) + np.square(x[0] - 1)


# first order derivatives of the function
def grad_rosenbrock(x):
    df1 = 400 * x[0] * (np.square(x[0]) - x[1]) + 2 * (x[0] - 1)
    df2 = -200 * (np.square(x[0]) - x[1])
    return np.array([df1, df2])


def inv_hess_rosenbrock(x):
    df11 = 1200 * np.square(x[0]) - 400 * x[1] + 2
    df12 = -400 * x[0]
    df21 = -400 * x[0]
    df22 = 200
    hess = np.array([[df11, df12], [df21, df22]])
    return inv(hess)


def newton(x, max_int):
    miter = 1
    step = .5
    vals = []
    objectfs = []
    # you can customize your own condition of convergence, here we limit the number of iterations
    while miter <= max_int:
        vals.append(x)
        objectfs.append(f_rosenbrock(x))
        temp = x - step * (inv_hess_rosenbrock(x).dot(grad_rosenbrock(x)))
        if np.abs(f_rosenbrock(temp) - f_rosenbrock(x)) > 0.01:
            x = temp
        else:
            break
        print(x, f_rosenbrock(x), miter)
        miter += 1
    return vals, objectfs, miter


start = [-1.2, 1]
val, objectf, iters = newton(start, 50)

x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objectf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='newton method')
fig.show()
fig.show()
