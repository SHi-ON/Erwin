# by Shayan Amani
# https://shayanamani.com


import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 4.5 * (1 - x) ** 2


alpha = np.linspace(-2, 3, 300)
y = f(alpha)

fig = plt.figure()

plt.scatter(alpha, y, s=1)

l_alpha = 0.5
r_alpha = 1.5
axe_max_l = f(l_alpha) / f(alpha[2])
axe_max_r = f(r_alpha) / f(alpha[0])
# it's weird that ymax and ymin do not follow the portion of the plot
# given by the fractions above
plt.axvline(x=1.5, ymin=0, ymax=0.05, linestyle='--', color='k', linewidth=1)
plt.text(1.5, -5, r'$\alpha$ = 1.5', rotation=90)
plt.axvline(x=0.5, ymin=0, ymax=0.05, linestyle='--', color='k', linewidth=1)
plt.text(0.5, -5, r'$\alpha$ = 0.5', rotation=90)

plt.axhline(y=f(l_alpha), linestyle='--', color='k', linewidth=1)

plt.xlabel(r'$\alpha$', labelpad=50, fontweight='extra bold')

plt.title(r'$\phi(\alpha) = f (x_k + \alpha_k p_k) $')

fig.show()
