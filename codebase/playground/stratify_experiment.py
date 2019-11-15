from sklearn.model_selection import train_test_split
import pandas as pd
import os


os.chdir('/Volumes/DevCamp/PyCharmProjects/Erwin/')

seed = 27

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

# always use stratify option to make sure about evenly distributed classes in test and train set
train, validate = train_test_split(raw_data, test_size=0.1, random_state=seed, stratify=raw_data['label'])
# train, validate = train_test_split(raw_data, test_size=0.1, random_state=seed)

train.head()
validate.head()
tr = train.groupby('label').count()

for i in range(9):
    ratio = tr.iloc[i][0] / cnt.iloc[i][0]
    print(ratio)
    assert 0.89 < ratio < 0.91, 'Ratio is not following the rules {}'.format(i)
