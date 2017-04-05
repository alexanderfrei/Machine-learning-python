import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def get_x(df):

    p_class = pd.get_dummies(df['Pclass'], prefix="Pclass")
    gender = pd.get_dummies(df['Sex'], prefix="Sex")
    embarked = pd.get_dummies(df['Embarked'], prefix="Embarked")

    imp_var = pd.DataFrame(imr.transform(df[['Age','Fare']]))
    imp_var.rename(columns={0: "Age", 1: "Fare"}, inplace=True)

    x = pd.DataFrame(pd.concat([p_class, gender, embarked, imp_var, df[['SibSp','Parch']]], axis=1))
    return x


gender_sub = pd.read_csv("data/gender_submission.csv", header=0)
train = pd.read_csv("data/train.csv", header=0)
test = pd.read_csv("data/test.csv", header=0)

imr = Imputer(missing_values='NaN', strategy='median', axis=0)
imr.fit_transform(train[["Age","Fare"]])

y_train = train['Survived']
X_train = get_x(train)
X_test = get_x(test)

scaler = StandardScaler()
std_X_train = scaler.fit_transform(X_train)
std_X_test = scaler.transform(X_test)

# logistic regression
lr = linear_model.LogisticRegression(C=0.5)
lr.fit(std_X_train, y_train)
train_proba = lr.predict_proba(std_X_train)
cutoff = optimal_cutoff(y_train, train_proba[:,1])

# predict
test_proba = lr.predict_proba(std_X_test)
prediction = np.where(test_proba[:,1] > cutoff, 1, 0)

submission = pd.concat([test['PassengerId'], pd.DataFrame(prediction.astype('int'))], axis=1)
submission.rename(columns={0: "Survived"}, inplace=True)
submission.to_csv("submission/lr1.csv", index=False)

# TODO 2) grid, pipeline


# print(pd.crosstab(gender_sub.Survived,gender_sub.Survived))
# df = pd.DataFrame({'col1':np.random.randn(100),'col2':np.random.randn(100)})
# df.hist(layout=(1,2))
#

# df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
#                     'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
# df4.diff().hist(alpha=0.5, color='k')
# plt.show()
