import seaborn as sns
import matplotlib.pyplot as plt

dd = sns.load_dataset("diamonds")

dd.head()

f = plt.figure(figsize=(10, 10))

sns.scatterplot(x="carat", y="price",
                alpha=0.5,
                data=dd)

plt.show()


