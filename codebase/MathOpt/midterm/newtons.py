import scipy.optimize as sp
import numpy as np
import rosenbrock


def f(x, y):
    return 0.5 * (4 * x ** 2 + y) ** 2 + 4 * y ** 2 - x ** 2 + x * y


x0 = np.array([0, 1])
x0.shape
type(x0)

min = sp.newton(rosenbrock.rb, x0)

if x0 >= 0:
    print('YES')
else:
    print('NO')


assert x0 > 0

f_0 = f(0, 1)
f_1 = f(-4.5, 0.5)
f_0
f_1