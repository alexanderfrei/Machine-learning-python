
# coding: utf-8
# example of hyperopt parameters tuning

from ml_tools.learning import *

K = 5

best_features_1 = ['ps_car_13',
                   'ps_reg_03',
                   'ps_ind_05_cat',
                   'ps_ind_03',
                   'ps_ind_15',
                   'ps_reg_02',
                   'ps_car_14',
                   'ps_car_12',
                   'ps_car_01_cat',
                   'ps_car_07_cat',
                   'ps_ind_17_bin',
                   'ps_car_03_cat',
                   'ps_reg_01',
                   'ps_car_15',
                   'ps_ind_01',
                   'ps_ind_16_bin',
                   'ps_ind_07_bin',
                   'ps_car_06_cat',
                   'ps_car_04_cat',
                   'ps_ind_06_bin',
                   'ps_car_09_cat',
                   'ps_car_02_cat',
                   'ps_ind_02_cat',
                   'ps_car_11',
                   'ps_car_05_cat',
                   'ps_calc_09',
                   'ps_calc_05',
                   'ps_ind_08_bin',
                   'ps_car_08_cat',
                   'ps_ind_09_bin',
                   'ps_ind_04_cat',
                   'ps_ind_18_bin',
                   'ps_ind_12_bin',
                   'ps_ind_14',
                   'ps_car_11_cat',
                   'missings',
                   'bin_diff',
                   'ps_reg_01_plus_ps_car_02_cat_cat',
                   'ps_reg_01_plus_ps_car_04_cat_cat',
                   'ps_car_11x14',
                   'ps_ind_01x_car_14']

def pickle_it(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def depickle_it(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

# In[2]:

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import warnings
warnings.filterwarnings('ignore')

# load

X, y, X_test = depickle_it('./input/X'), depickle_it('./input/y'), depickle_it('./input/X_test')
id_train, id_test = depickle_it('./input/id_train'), depickle_it('./input/id_test')

X = X[best_features_1]
X_test = X_test[best_features_1]

# model

model = XGBClassifier(n_jobs=4, tree_method='gpu_hist', predictor="cpu_predictor", objective="binary:logistic",
                      n_estimators=600,
                      learning_rate=0.02,
                      max_depth=5,
                      gamma=4, min_child_weight=5,
                      subsample=.8, colsample_bytree=.8,
                      scale_pos_weight=1.5,
                      #                         reg_alpha=5,
                      #                         reg_lambda=1.3,
                      )

kf = KFold(n_splits=K, random_state=42, shuffle=True)

# hyperopt

from hyperopt import hp
from hyperopt import fmin, tpe, Trials
import os

if os.path.isfile('trials'):
    trials = depickle_it('trials')
else:
    trials = Trials()

space = {}
space['n_estimators'] = hp.randint('n_estimators', 20) * 30 + 400
space['max_depth'] = hp.choice('max_depth', [4, 5, 6])
space['learning_rate'] = hp.uniform('learning_rate', 0.005, 0.05)
space['gamma'] = hp.randint('gamma', 6) * 1.5
space['min_child_weight'] = hp.randint('min_child_weight', 10) + 1
space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 1, 2)
space['subsample'] = hp.uniform('subsample', 0.6, 0.9)
space['colsample_bytree'] = hp.uniform('colsample_bytree', 0.6, 0.9)
# hp.quniform('eta', 0.07,0.2, 0.001) - квантификация hyperopt
# space['reg_alpha'] = hp.randint('reg_alpha', 8)*2
# space['reg_lambda'] = hp.randint('reg_lambda', 8)*2


def objective(params):
    i = len(trials.trials)
    print("*" * 50 + "\niter = {}".format(i))
    gc.collect()
    model.set_params(**params)

    _, _, _, gini_cv = cv_learn(X, y, X_test, model, False, kf, True, True, min_samples_leaf=100, smoothing=5)

    print("gini = {}".format(gini_cv))
    pickle_it(trials, 'trials')
    i += 1

    return -gini_cv


best = fmin(objective,
            space,
            algo=tpe.suggest,
            max_evals=125,
            trials=trials)

pickle_it(trials, 'trials')
pickle_it(best, 'best')
