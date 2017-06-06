
from sklearn.model_selection import StratifiedKFold  # save proportions of classes in folds
from sklearn.model_selection import KFold  # random folds
import pandas as pd
import numpy as np


def folding(y, X, eval_size, stratified=True):

    """ Folding set into validation and train sets
    :param y: target variable
    :param X: features
    :param eval_size: ratio of validation fold
    :param stratified: True for classification folding
    :return: 4 sets: 2 train -> 2 valid

    Example:
    X_train, y_train, X_valid, y_valid = folding(y, X, 0.1, True)

    """

    if stratified:
        kf = StratifiedKFold(y, round(1 / eval_size))
    else:
        kf = KFold(len(y), round(1 / eval_size))

    train_indices, valid_indices = next(iter(kf))

    if isinstance(X, pd.DataFrame):
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]  # pandas dataframe
        X_valid, y_valid = X.iloc[valid_indices], y.iloc[valid_indices]
    elif isinstance(X, (np.ndarray, np.generic)):
        X_train, y_train = X[train_indices], y[train_indices]  # numpy ndarray
        X_valid, y_valid = X[valid_indices], y[valid_indices]
    else:
        raise TypeError("Convert data to pandas dataframe or numpy ndarray")

    return X_train, y_train, X_valid, y_valid


#  TODO: check all

#
#
#
#
# def xgb_fit(alg, X, y, score='auc', cv=5, early_stopping=50, show_feat=False):
#     """
#     :param alg: Xgboost binary classifier
#     :param show_feat: Show feature importance (bool)
#     :return: None
#     1. Fit number of trees with cv
#     2. Print accuracy and auc score
#     3. Show feature importance
#     """
#
#     import xgboost as xgb
#     from sklearn import metrics
#     import matplotlib.pyplot as plt
#
#     xg_train = xgb.DMatrix(X, label=y)
#     cv_result = xgb.cv(alg.get_xgb_params(), xg_train,
#                        num_boost_round=alg.get_params()['n_estimators'], nfold=cv,
#                        metrics=score, early_stopping_rounds=early_stopping)
#
#     alg.set_params(n_estimators=cv_result.shape[0])
#     alg.fit(X, y, eval_metric=score)
#
#     predictions = alg.predict(X)
#     pred_prob = alg.predict_proba(X)[:, 1]
#
#     print(alg)
#     print("Accuracy : %.4g" % metrics.accuracy_score(y, predictions))
#     print("AUC Score (Train): %f" % metrics.roc_auc_score(y, pred_prob))
#
#     if show_feat:
#         feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#         feat_imp.plot(kind='bar', title='Feature Importances')
#         plt.ylabel('Feature Importance Score')
#         plt.show()
#
#
# def grid_search(estimator, params, X, y, scoring='roc_auc', cv=5):
#     """
#     Grid Search wrapper with results print
#     :return: best estimator
#     """
#
#     from sklearn.model_selection import GridSearchCV
#
#     gs = GridSearchCV(estimator=estimator,
#                       param_grid=params,
#                       scoring=scoring,
#                       cv=cv, n_jobs=-1,
#                       verbose=2)
#
#     gs = gs.fit(X, y)
#
#     [print("{}, mean: {}, std: {} ".format(param, score, std)) for (score, std, param) in zip(
#         gs.cv_results_['mean_test_score'],
#         gs.cv_results_['std_test_score'],
#         gs.cv_results_['params'])]
#
#     print("Best params: {} \nBest score: {}".format(gs.best_params_, gs.best_score_))
#     return gs.best_estimator_
#
#
# def get_oof(clf, x_train, y_train, x_test, n_splits=5):
#     """
#     Out-of-fold learning
#     Prevent overfitting
#     :param clf: estimator
#     :param x_train: ndarray
#     :param y_train: ndarray
#     :param x_test: ndarray
#     :param n_splits: number of splits
#     :return: oof_train (prediction by cv, 0/1 for binary), oof_test (prediction within ALL cv iterations,
#     by majority vote)
#     """
#
#     from sklearn.model_selection import KFold
#     import numpy as np
#
#     n_train = x_train.shape[0]
#     n_test = x_test.shape[0]
#     kf = KFold(n_splits)
#
#     oof_train = np.zeros((n_train,))
#     oof_test = np.zeros((n_test,))
#     oof_test_skf = np.empty((n_splits, n_test))
#
#     i = 0
#     for train_index, test_index in kf.split(x_train):
#         x_tr, x_te = x_train[train_index], x_train[test_index]
#         y_tr = y_train[train_index]
#         clf.fit(x_tr, y_tr)
#         oof_train[test_index] = clf.predict(x_te)
#         oof_test_skf[i, :] = clf.predict(x_test)
#         i += 1
#
#     oof_test[:] = oof_test_skf.mean(axis=0)
#     print("_ OOF learned _")
#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#
#
# def fit_itera(clf, params, x_train, y_train, test, n_iter):
#     """
#     majority vote for classification
#     :param clf: classifier
#     :param n_iter: number of fit/predict iterations
#     :return: vector of consensus prediction
#     """
#     import numpy as np
#     pred = np.zeros([test.shape[0], n_iter])
#     for i in range(n_iter):
#         rf = clf(**params, random_state=None)
#         rf.fit(x_train, y_train)
#         print("%.4f" % rf.oob_score_)
#         pred[:, i] = rf.predict(test)
#     return pred.mean(axis=1).round().astype(int)