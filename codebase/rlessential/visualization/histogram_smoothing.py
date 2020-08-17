import matplotlib.pyplot as plt
import numpy as np

a = np.random.normal(5, 1, 1000) * 10
a = a.astype(int)

# slides theme color
bar_color = '#0C2B36'

fig, ax = plt.subplots()
ax.set_xticks([], [])
ax.set_yticks([], [])
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(221)
plt.hist(a,
         bins=5,
         density=False,
         align='mid',
         color=bar_color,
         edgecolor='gray')
ax1.set_title('Oversmoothed')

ax2 = fig.add_subplot(222)
plt.hist(a,
         bins=60,
         density=False,
         align='left',
         color=bar_color,
         edgecolor='gray')
ax2.set_title('Undersmoothed')

fig.add_gridspec(2, 1)

ax3 = fig.add_subplot(212)
plt.hist(a,
         bins='fd',
         density=False,
         align='left',
         color=bar_color,
         edgecolor='gray')
ax3.set_title('Just Right')

fig.tight_layout()


plt.savefig('over-under-smoothed.pdf', format='pdf')
plt.show()
