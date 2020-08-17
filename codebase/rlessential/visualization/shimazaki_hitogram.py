import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, size, zeros
from numpy.random import normal
from scipy import linspace

# slides theme colors
theme_blue = '#0C2B36'
theme_red = '#E04D4F'

x = normal(4, 1, 100)  # Generate n pseudo-random numbers with (mu,sigma,n)
x = [4.37, 3.87, 4.00, 4.03, 3.50, 4.08, 2.25, 4.70, 1.73, 4.93, 1.73, 4.62, \
     3.43, 4.25, 1.68, 3.92, 3.68, 3.10, 4.03, 1.77, 4.08, 1.75, 3.20, 1.85, \
     4.62, 1.97, 4.50, 3.92, 4.35, 2.33, 3.83, 1.88, 4.60, 1.80, 4.73, 1.77, \
     4.57, 1.85, 3.52, 4.00, 3.70, 3.72, 4.25, 3.58, 3.80, 3.77, 3.75, 2.50, \
     4.50, 4.10, 3.70, 3.80, 3.43, 4.00, 2.27, 4.40, 4.05, 4.25, 3.33, 2.00, \
     4.33, 2.93, 4.58, 1.90, 3.58, 3.73, 3.73, 1.82, 4.63, 3.50, 4.00, 3.67, \
     1.67, 4.60, 1.67, 4.00, 1.80, 4.42, 1.90, 4.63, 2.93, 3.50, 1.97, 4.28, \
     1.83, 4.13, 1.83, 4.65, 4.20, 3.93, 4.33, 1.83, 4.53, 2.03, 4.18, 4.43, \
     4.07, 4.13, 3.95, 4.10, 2.27, 4.58, 1.90, 4.50, 1.95, 4.83, 4.12]

x_max = max(x)
x_min = min(x)

N_MIN = 4  # Minimum number of bins (integer) must be more than 1 (N_MIN > 1)
N_MAX = 50  # Maximum number of bins (integer)
N = np.arange(N_MIN, N_MAX)

W = (x_max - x_min) / N  # Bin width vector

C = zeros(shape=(size(W), 1))

# Computation of the cost function
for i in range(size(N)):
    edges = linspace(x_min, x_max, N[i] + 1)  # Bin edges
    ki = plt.hist(x, edges)  # Count # of events in bins
    ki = ki[0]
    k = mean(ki)  # Mean of event count
    v = sum((ki - k) ** 2) / N[i]  # Variance of event count
    # v = np.var(ki)
    C[i] = (2 * k - v) / ((W[i]) ** 2)  # The cost Function

# Optimal Bin Size Selection
C_min = min(C)
index_optimal = np.argmin(C)
W_optimal = W[index_optimal]

# plot histogram with the optimal bin-width
edges = linspace(x_min, x_max, N[index_optimal] + 1)
plt.hist(x, edges, color=theme_blue, edgecolor='gray')
plt.title(u"Histogram")
plt.ylabel(u"Frequency")
plt.xlabel(u"Value")
plt.savefig('histogram_shimazaki.pdf', format='pdf')
plt.show()

plt.plot(W, C, '-', color='gray', lw=1)
plt.scatter(W, C, marker='.', c=theme_blue, sizes=[70.0])
plt.scatter(W_optimal, C_min, marker='*', c=theme_red, sizes=[70.0])
plt.title(u"Cost Function $C(w)$")
plt.ylabel(u"Loss")
plt.xlabel(u"Bin-widths")
plt.savefig('cost_shimazaki.pdf', format='pdf')
plt.show()
