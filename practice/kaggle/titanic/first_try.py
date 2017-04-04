import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

gender_sub = pd.read_csv("data/gender_submission.csv", header=0)
train = pd.read_csv("data/train.csv", header=0)
test = pd.read_csv("data/test.csv", header=0)


def get_x(df):

    p_class = pd.get_dummies(df['Pclass'], prefix="Pclass")
    gender = pd.get_dummies(df['Sex'], prefix="Sex")
    embarked = pd.get_dummies(df['Embarked'], prefix="Embarked")

    imr = Imputer(missing_values='NaN', strategy='median', axis=0)
    imr.fit(df["Age"].values.reshape(-1, 1))
    age = pd.DataFrame(imr.transform(df['Age'].values.reshape(-1, 1)))
    age.rename(columns={0: "Age"}, inplace=True)

    x = pd.DataFrame(pd.concat([p_class, gender, embarked, age, df[['SibSp','Parch','Fare']]], axis=1)).head(10)
    return x

y_train = train['Survived']
X_train = get_x(train)
X_test = get_x(test)

lr = linear_model.LogisticRegression(C=0.1)
# TODO 1)std 2)lr fit (C=1000,1,0.1) 3) grid, pipeline



# print(pd.crosstab(gender_sub.Survived,gender_sub.Survived))
# df = pd.DataFrame({'col1':np.random.randn(100),'col2':np.random.randn(100)})
# df.hist(layout=(1,2))
#

# df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
#                     'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
# df4.diff().hist(alpha=0.5, color='k')
# plt.show()




