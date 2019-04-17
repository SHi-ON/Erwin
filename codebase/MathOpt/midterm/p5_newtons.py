import scipy.optimize as sp

import rosenbrock
from mid_util import *

x0 = np.array([0, 1])
x0.shape
type(x0)

# FIXME
min = sp.newton(rosenbrock.rb, x0)

if x0 >= 0:
    print('YES')
else:
    print('NO')

assert x0 > 0

f_0 = f(np.array([0, 1]))
f_1 = f(np.array([-4.5, 0.5]))
f_0
f_1
