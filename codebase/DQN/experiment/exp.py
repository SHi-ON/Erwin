import matplotlib.pyplot as plt
import numpy as np

for i in range(100):
    n = i * np.random.randint(0, 6)
    # plt.scatter(n, i)
    plt.plot(n, i, linestyle='-', marker="o")
plt.ylabel('Loss over time of training')
plt.show()







# library and dataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Create data
df = pd.DataFrame({'x': range(1, 101), 'y': np.random.randn(100) * 15 + range(1, 101),
                   'z': (np.random.randn(100) * 15 + range(1, 101)) * 2})

# plot with matplotlib
plt.plot('x', 'y', data=df, marker='o', color='mediumvioletred')
plt.show()

# Just load seaborn and the chart looks better:
import seaborn as sns

plt.plot('x', 'y', data=df, marker='o', color='mediumvioletred')
plt.show()

