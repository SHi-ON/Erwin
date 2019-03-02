import numpy as np
from numpy import linalg as la
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from rosenbrock import *


def f_rosenbrock(a, b):
    return 100 * (b - a ** 2) ** 2 + (1 - a) ** 2


if __name__ == "__main__":

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = rb(np.array([X, Y]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.contour3D(X, Y, Z, 200, cmap='magma')

    ax.view_init(60, 35)
    fig.show()
    fig.show()

    # min finding
    min_func = np.min(f_rosenbrock(X, Y))
    print("min=", min_func)

    # eigenvalues
    A = np.array([[802, -400], [-400, 200]])
    eig_val, eig_vec = la.eig(A)
    print("eigenvalues:", eig_val, la.eigvals(A))
    print("eigenvectors:", eig_vec)

    # gradient
    grad = np.gradient(f_rosenbrock(X, Y))
    grad


