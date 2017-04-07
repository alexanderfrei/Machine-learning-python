import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

# window parameters

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# functions


def get_x(df):

    # one hot recoding of class is mistake - we lose part of information
    # p_class = pd.get_dummies(df['Pclass'], prefix="Pclass")
    p_class = df['Pclass']

    gender = pd.get_dummies(df['Sex'], prefix="Sex")
    embarked = pd.get_dummies(df['Embarked'], prefix="Embarked")

    imp_var = pd.DataFrame(imr.fit_transform(df[['Age','Fare']]))
    imp_var.rename(columns={0: "Age", 1: "Fare"}, inplace=True)

    x = pd.DataFrame(pd.concat([p_class, gender, embarked, imp_var, df[['SibSp','Parch']]], axis=1))
    return x

# X, y, predict_data
gender_sub = pd.read_csv("data/gender_submission.csv", header=0)
predict_data = pd.read_csv("data/test.csv", header=0)
train_data = pd.read_csv("data/train.csv", header=0)

# init fitting objects
imr = Imputer(missing_values='NaN', strategy='median', axis=0)
scaler = StandardScaler()

# preprocessing
X_train, y_train = get_x(train_data.drop("Survived", axis=1)), train_data['Survived']

# train/test split
# X_train, X_test, y_train, y_test = \
#     train_test_split(X_train, y_train, test_size=0.20)

# pipeline logistic regression train
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(C=0.075))])

# train/test score
pipe_lr.fit(X_train, y_train)
# print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# cv
# scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# learning curves
train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.75, 0.85])
plt.tight_layout()
# plt.savefig('./figures/learning_curve.png', dpi=300)
plt.show()

# validation parameter curve
#
# param_range = [0.05, 0.075, 0.1, 0.125, 0.15]
# train_scores, test_scores = validation_curve(
#                 estimator=pipe_lr,
#                 X=X_train,
#                 y=y_train,
#                 param_name='clf__C',
#                 param_range=param_range,
#                 cv=10)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.plot(param_range, train_mean,
#          color='blue', marker='o',
#          markersize=5, label='training accuracy')
#
# plt.fill_between(param_range, train_mean + train_std,
#                  train_mean - train_std, alpha=0.15,
#                  color='blue')
#
# plt.plot(param_range, test_mean,
#          color='green', linestyle='--',
#          marker='s', markersize=5,
#          label='validation accuracy')
#
# plt.fill_between(param_range,
#                  test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
#
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Parameter C')
# plt.ylabel('Accuracy')
# plt.ylim([0.75, 0.85])
# plt.tight_layout()
# # plt.savefig('./figures/validation_curve.png', dpi=300)
# plt.show()


# gridSearch

# param_range = np.linspace(0.05,0.15,20)
# param_grid = [{'clf__C': param_range}]
# gs = GridSearchCV(estimator=pipe_lr,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=10)
# gs = gs.fit(X_train, y_train)
# pipe_lr = gs.best_estimator_

# predict and submit

# X_predict = get_x(predict_data)
# y_pred = pipe_lr.predict(X_predict)
#
# submission = pd.concat([predict_data['PassengerId'], pd.DataFrame(y_pred.astype('int'))], axis=1)
# submission.rename(columns={0: "Survived"}, inplace=True)
# submission.to_csv("submission/lr5.csv", index=False)


# TODO 3) random forest + importance 4) xgboost 5) solutions


#### exploration

# print(pd.crosstab(gender_sub.Survived,gender_sub.Survived))
# df = pd.DataFrame({'col1':np.random.randn(100),'col2':np.random.randn(100)})
# df.hist(layout=(1,2))
#

# df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
#                     'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
# df4.diff().hist(alpha=0.5, color='k')
# plt.show()


#### compare test results of 2 submissions

# cd C:\Users\Aleksandr.Turutin\PycharmProjects\python_machine_learning\practice\kaggle\titanic\submission
# df1 = pd.read_csv("lr3.csv", header=0)
# df2 = pd.read_csv("lr5.csv", header=0)
# df1.rename(columns={'Survived': 1}, inplace=True)
# df2.rename(columns={'Survived': 2}, inplace=True)
# df = pd.concat([df1[1], df2[2]], axis=1)
# pd.crosstab(df[1],df[2])
