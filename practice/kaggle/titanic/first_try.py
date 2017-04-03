import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# os.getcwd()

gender_sub = pd.read_csv("data/gender_submission.csv", header=0)
train = pd.read_csv("data/train.csv", header=0)
test = pd.read_csv("data/test.csv", header=0)

y_train = train['Survived']
# X_train = train

ohe = OneHotEncoder(categorical_features='all', handle_unknown='error', n_values='auto', sparse=True)
p_class = ohe.fit_transform(train['Pclass'].values.reshape(-1,1)).toarray()
# print(p_class)

lr = linear_model.LinearRegression()

print(train)


# print(pd.crosstab(gender_sub.Survived,gender_sub.Survived))
# df = pd.DataFrame({'col1':np.random.randn(100),'col2':np.random.randn(100)})
# df.hist(layout=(1,2))
#

# df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
#                     'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
# df4.diff().hist(alpha=0.5, color='k')
# plt.show()












