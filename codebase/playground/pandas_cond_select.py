import seaborn as sns

iris = sns.load_dataset('iris')
iris.head()

iris

rd = iris.loc[iris['sepal_length'] == 4.9]
rd
