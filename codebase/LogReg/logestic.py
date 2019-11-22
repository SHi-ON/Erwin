from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics # confusion matrix
import pandas as pd
import os


os.chdir('/Volumes/DevCamp/PyCharmProjects/Erwin/')

seed = 42

raw_data = pd.read_csv("codebase/dataset/mnist/train.csv")
raw_data.count()
raw_data['label'].count()
raw_data['pixel9'].count()

cnt = raw_data.groupby('label').count()
cnt
type(raw_data.groupby('label'))
type(cnt)

cnt.iloc[1].sum()
cnt['pixel0'].sum()

cnt = raw_data.groupby('label').count()

raw_data.columns
X = raw_data.loc[:, raw_data.columns != 'label']
X.columns

y = raw_data.loc[:, ['label']]  # returns DataFrame
y = raw_data['label']  # returns Series
y

# always use stratify option to make sure about evenly distributed classes in test and train set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed, stratify=raw_data['label'])
# train, validate = train_test_split(raw_data, test_size=0.1, random_state=seed)


# train.head()
# validate.head()
# tr = train.groupby('label').count()
#
# for i in range(9):
#     ratio = tr.iloc[i][0] / cnt.iloc[i][0]
#     print(ratio)
#     assert 0.89 < ratio < 0.91, 'Ratio is not following the rules {}'.format(i)


# log reg
log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

conf_mat = metrics.confusion_matrix(y_test, y_pred)

