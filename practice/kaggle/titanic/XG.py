import os
from practice.kaggle.titanic.FE import *

sub = 'submission/xg3.csv'

train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))

train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin_num(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = ticket_grouped(train, test)
train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = drop(train, test)

replace_cabin(train)
replace_cabin(test)

####################################################################################################
#

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# gbm = XGBClassifier(silent=False)
#
# param_grid = {'max_depth': np.linspace(3,9,4).astype(int),
#               'min_child_weight': np.linspace(1,5,3),
#               'learning_rate': np.linspace(0.1, 0.9, 4),
#               'n_estimators': [30,50]}

# gs = GridSearchCV(estimator=gbm,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=10)
#
# gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])
# print(gs.best_estimator_, gs.best_score_)
# xbm_best = gs.best_estimator_

xbm_best = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.10000000000000001, max_delta_step=0,
       max_depth=9, min_child_weight=1.0, missing=None, n_estimators=50,
       nthread=-1, objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=False, subsample=1)

xbm_best.fit(train.iloc[:, 1:], train.iloc[:, 0])

# ################################################################################################
# # learning curves
#
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
#
# train_sizes, train_scores, test_scores =\
#                 learning_curve(estimator=xbm_best,
#                                X=train.iloc[:, 1:],
#                                y=train.iloc[:, 0],
#                                train_sizes=np.linspace(0.1, 1.0, 10),
#                                scoring="accuracy",
#                                cv=10,
#                                n_jobs=1)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.plot(train_sizes, train_mean,
#          color='blue', marker='o',
#          markersize=5, label='training accuracy')
#
# plt.fill_between(train_sizes,
#                  train_mean + train_std,
#                  train_mean - train_std,
#                  alpha=0.15, color='blue')
#
# plt.plot(train_sizes, test_mean,
#          color='green', linestyle='--',
#          marker='s', markersize=5,
#          label='validation accuracy')
#
# plt.fill_between(train_sizes,
#                  test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
#
# plt.grid()
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.7, 1])
# plt.tight_layout()
# plt.show()
# ################################################################################################


####################################################################################################

predictions = xbm_best.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
test = pd.read_csv(os.path.join('data', 'test.csv'))
predictions = pd.concat((test.iloc[:, 0], predictions), axis=1)
predictions.to_csv(sub, sep=",", index=False)

