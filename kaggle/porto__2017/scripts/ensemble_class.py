from ml_tools.metrics import *
from ml_tools.tools import *
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        """
        :param n_splits: n splits for cross-validation  
        :param stacker: estimator for stacking (typically logreg) 
        :param base_models: tuple of estimators 
        """
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):

        """
        :param X: train 
        :param y: labels 
        :param T: test  
        :return: tuple of prediction for train and test sets  
        """

        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        val_pred = np.zeros(y.shape[0], dtype=np.float32)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=10101).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                print("Fit %s fold %d" % (str(clf).split('(')[0], j + 1))

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:, 1]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]

            S_test[:, i] = S_test_i.mean(axis=1)

        # test predict
        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:, 1]

        # cv train replace
        folds = list(KFold(n_splits=10, shuffle=True, random_state=1).split(S_train, y))
        for i, (train_idx, test_idx) in enumerate(folds):
            X_train = S_train[train_idx]
            y_train = y[train_idx]
            X_holdout = S_train[test_idx]

            self.stacker.fit(X_train, y_train)
            val_pred[test_idx] = self.stacker.predict_proba(X_holdout)[:, 1]

        print("Gini: {}".format(eval_gini(y, val_pred)))

        return val_pred, res

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print("run preprocessing..")

# simple preprocessing
id_test = test['id'].values
id_train = train['id'].values
target_train = train['target'].values
train = train.drop(['target', 'id'], axis=1)
test = test.drop(['id'], axis=1)

# drop "calc" features
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)

# replace -1 with nans
train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

# one hot encoding
cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]))
    train = pd.concat([train, temp], axis=1)
    train = train.drop([column], axis=1)

for column in cat_features:
    temp = pd.get_dummies(pd.Series(test[column]))
    test = pd.concat([test, temp], axis=1)
    test = test.drop([column], axis=1)

print(train.values.shape, test.values.shape)

#base models



# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8
lgb_params['min_child_samples'] = 500
lgb_params['seed'] = 99

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 99

lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 99

# rf_model = RandomForestClassifier(**rf_params)
# et_model = ExtraTreesClassifier(**et_params)
# xgb_model = XGBClassifier(**xgb_params)
# cat_model = CatBoostClassifier(**cat_params)
# rgf_model = RGFClassifier(**rgf_params)
# gb_model = GradientBoostingClassifier(max_depth=5)
# ada_model = AdaBoostClassifier()

lgb_model = LGBMClassifier(**lgb_params)
lgb_model2 = LGBMClassifier(**lgb_params2)
lgb_model3 = LGBMClassifier(**lgb_params3)
log_model = LogisticRegression()


# stacker
print("run stacking..")
stack = Ensemble(n_splits=5,
                 stacker=log_model,
                 base_models=(lgb_model, lgb_model2, lgb_model3))

y_val_pred, y_pred = stack.fit_predict(train, target_train, test)

print(y_val_pred.shape, y_pred.shape)


# save results
save_sub('../train/lightgbm_stacker.csv', id_train, y_val_pred, 'id', 'target')
save_sub('../test/lightgbm_stacker.csv', id_test, y_pred, 'id', 'target')
