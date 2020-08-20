import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

theme_blue = '#0C2B36'
theme_red = '#E04D4F'
theme_green = '#00F900'

mu = 5
sig = 1

# Generate some data for this demonstration.
data = norm.rvs(10.0, 2.5, size=500)

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=9, density=True, alpha=0.8, color=theme_blue, edgecolor='gray')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, theme_red, linewidth=4)

plt.gca().spines['left'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks([], [])
plt.yticks([], [])

# plt.savefig('dist_initial.svg', format='svg')
plt.savefig('dist_final.svg', format='svg')

plt.show()
