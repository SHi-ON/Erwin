import numpy as np
from numpy import linalg as la
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from rosenbrock import *


def f(a, b):
    return a * b


if __name__ == "__main__":

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.contour3D(X, Y, Z, 200, cmap='magma')

    c1 = plt.Circle((0, 0), 0.0040, color='green', fill=False)
    ax.add_artist(c1)

    ax.view_init(40, 30)
    fig.show()
    fig.show()

