# coding: utf-8

import numpy as np
import pandas as pd
import gc, pickle, os

from hyperopt import hp
from hyperopt import fmin, tpe, Trials

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
 
################################################################### space ######################################################################

space = {
    'verbose': 0,
    'num_threads': 4,
    'objective': 'binary',
    'metric': 'auc',
    'subsample_freq': 1,

    'learning_rate': 0.3,
    'scale_pos_weight': 300,
    
    'num_leaves': 7,
    'max_depth': 3,
    'min_data_in_leaf': 500,    
    'min_sum_hessian_in_leaf': .0,   
    'min_split_gain': .0,
    'lambda_l1': .0,
    'lambda_l2': .0, 

    'max_bin': 255,
    'subsample_for_bin': 200000,
    'subsample': 0.8,
    'colsample_bytree': 1,    
}

space['learning_rate'] = hp.choice('learning_rate', [0.05, 0.1, 0.15, 0.2, 0.25])
space['max_depth'] = hp.choice('max_depth', [0, 3, 4, 5, 6])
space['num_leaves'] = hp.randint('num_leaves', 12)*2 + 3

space['min_data_in_leaf'] = hp.randint('min_data_in_leaf', 5) * 1000 + 500
space['min_split_gain'] = hp.quniform('min_split_gain', 0, 3, 0.25)
space['min_sum_hessian_in_leaf'] = hp.quniform('min_sum_hessian_in_leaf', 0, 0.16, 0.02)
space['lambda_l1'] = hp.quniform('lambda_l1', 0, 3, 0.5)
space['lambda_l2'] = hp.quniform('lambda_l2', 0, 3, 0.5)

space['scale_pos_weight'] = hp.choice('scale_pos_weight', [200, 300, 400, 500])
space['max_bin'] = hp.choice('max_bin', [63, 100, 170, 255])
space['subsample_for_bin'] = hp.randint('subsample_for_bin', 3) * 200000 + 200000
space['subsample'] = hp.choice('subsample', [0.6, 0.7, 0.8, 0.9])
space['colsample_bytree'] = hp.choice('colsample_bytree', [0.8, 0.9, 1])

with open('space.pkl', 'wb') as f: pickle.dump(obj=space, file=f)

################################################################### load ######################################################################
print('Load files..')

if os.path.isfile('trials.pkl'):
    with open('trials.pkl', 'rb') as f: trials = pickle.load(f)
else:
    trials = Trials()

if os.path.isfile('max_score.pkl'): 
    with open('max_score.pkl', 'rb') as f: max_score = pickle.load(f)
else:
    max_score = [0]

skiprows = 111886952
nrows = 184903889
ntrain = 131886952
nval = nrows - ntrain

nrows =  30000000
ntrain = 20000000

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint8',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

train = pd.read_csv("../input/train.csv", parse_dates=['click_time'], dtype=dtypes, 
                    skiprows=[1, skiprows], nrows=nrows, 
                    usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('total: {}, train: {}, valid: {}'.format(str(train.shape[0]), str(ntrain), str(nval)))

################################################################### functions ######################################################################

def next_click(df):
    df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    df['nextClick'] = (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time).astype(np.float32)

def count(df, group_cols, agg_name, agg_type='uint32'):
    print( "Aggregating by ", group_cols , '...' )
    agg = df[group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df[agg_name] = df.merge(agg, on=group_cols, how='left')[agg_name].astype(agg_type)
    
def preprocessing(df, df_name, to_drop):
    
    print('\npreprocess', df_name)
    
    # temporal
    df['hour'] = df.click_time.dt.hour
    next_click(df)
    
    # counts 
    count(df, ['ip'], 'ip_count')
    count(df, ['ip', 'app'], 'app_ch_count')
    
    # drop 
    df.drop(to_drop, 1, inplace=True)

################################################################### preprocessing ######################################################################
print('Preprocessing..')

y = train['is_attributed']
X_train, X_valid = train[:ntrain].copy(), train[ntrain:ntrain+nval].copy(),
y_train, y_valid = y[:ntrain], y[ntrain:ntrain+nval]
 
# X_test, y_test = train[ntrain+nval:].copy(), y[ntrain+nval:]
del train
gc.collect()

to_drop = ['ip', 'click_time', 'is_attributed']

preprocessing(X_train, 'train', to_drop)
preprocessing(X_valid, 'valid', to_drop)
# preprocessing(X_test, 'test', to_drop)

categorical = ['app', 'channel', 'device', 'os', 'hour']

predictors = X_train.columns.tolist()

X_train = X_train.values.astype(np.int32)
X_valid = X_valid.values.astype(np.int32)


lg_train = lgb.Dataset(X_train, label=y_train,
                       feature_name=predictors, categorical_feature=categorical, free_raw_data=False )
lg_val = lgb.Dataset(X_valid, label=y_valid,
                     feature_name=predictors, categorical_feature=categorical, free_raw_data=False )

del X_train, X_valid
gc.collect()

###################################################################### objective ######################################################################

def objective(params):

    print("*" * 100 + "\niter = {}".format(len(trials.trials)))
    print(params)
    gc.collect()

    model = lgb.train(params, lg_train, 
                     valid_sets=[lg_val], 
                     valid_names=['valid'], 
                     num_boost_round=1500,
                     early_stopping_rounds=20,
                     verbose_eval=20)

    score = model.best_score['valid']['auc']
    if score > max_score[-1]: 
        max_score.append(score)
        print('max score updated:', score)
    else:
        print(score)

    with open('trials.pkl', 'wb') as f: pickle.dump(obj=trials, file=f)
    with open('max_score.pkl', 'wb') as f: pickle.dump(obj=max_score, file=f)

    return -score

######################################################################### opt #################################################################

best = fmin(objective,
            space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

with open('best.pkl', 'wb') as f: pickle.dump(obj=best, file=f)
