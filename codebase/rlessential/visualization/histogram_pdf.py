import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# slides theme color
bar_color = '#0C2B36'

a = np.random.normal(5, 1, 1000) * 10
a = a.astype(int)

# density plot and histogram
sns.distplot(a,
             hist=True, kde=True,
             bins='fd', color=bar_color,
             hist_kws={'edgecolor': 'gray'},
             kde_kws={'linewidth': 4})

plt.savefig('histogram_pdf.pdf', format='pdf')
plt.show()
