
# coding: utf-8



def cross_val(X, y, model, kf, preprocessing, verbose=False):
    
    X, y = np.array(X), np.array(y).reshape(-1)
    cv_scores = []

    for i, (train_index, val_index) in enumerate(kf.split(X, y)):

        print( "Fold ", i)

        y_train, y_val = y[train_index].copy(), y[val_index].copy()
        X_train, X_val = X[train_index, :].copy(), X[val_index, :].copy()
        
        X_train, X_val = preprocessing(X_train, X_val, y_train)
        # if verbose: print(X_train.shape)
            
        fit_model = model.fit(X_train, y_train)
        pred = fit_model.predict_proba(X_val)[:, 1]
        
        cv_scores.append(roc_auc_score(y_val, pred))
        
    if verbose: print(np.mean(cv_scores))
    return cv_scores


def transform(train, test, target, to_drop=False, 
              high_cardinality="smoothing", hc_treshold = 10, hc_drop=False, # high cardinality categorical
              eb_k=50, eb_f=10,  # parameters for hc smoothing function 
              encode=False,  # categorical 
              fill_num=-1, scaling=False,  # continuous 
              feature_names = False, feature_dtypes = False,
              keep = False
             ):
    
    """ 
    data preprocessing 
    
    :train, test: pandas DataFrame
    :high_cardinality: way to handle categorical features with high number of levels
    :encode: category encoding, 'ohe' = one hot, 'bin' = binary
    :fill_num: fill nan for continuous features, -1 = with -1, ('mean', 'median') = strategy
    :scaling: 'standard' = StandartScaler
    :feature_names: list, columns from dataframe
    :feature_dtypes: pd.Series, dtypes from dataframe
    
    category features should have type 'object'
    """
    
    # checks 
    
    feature_names = list(feature_names)
    if (isinstance(train, np.ndarray) or isinstance(test, np.ndarray)) and not feature_names:
        raise FeatureNamesError("Feature names had not defined for np.ndarray")
    
    # np to pd 
    if isinstance(train, np.ndarray):
        train=pd.DataFrame(train, columns=feature_names)
        dt = feature_dtypes.to_dict()
        for col in train.columns:
            train[col] = train[col].astype(dt[col])        
    if isinstance(test, np.ndarray):
        test=pd.DataFrame(test, columns=feature_names)
        dt = feature_dtypes.to_dict()
        for col in train.columns:
            test[col] = test[col].astype(dt[col])
    
    # remove duplicates 
    if to_drop:
        train = train.drop(to_drop, axis=1)
        test = test.drop(to_drop, axis=1)
    
    ######## categorical features 
    
    cat_features = train.columns[train.dtypes=='object']
    num_features = train.columns[train.dtypes!='object']      
    
    # factorize 
    le = LabelEncoder()
    train[cat_features] = train[cat_features].fillna('-1')
    test[cat_features] = test[cat_features].fillna('-1')
    for c in cat_features:
        data=train[c].append(test[c])
        le.fit(data.values.tolist())  # nan = 0 level
        train[c] = le.transform(train[c].values.tolist())
        test[c] = le.transform(test[c].values.tolist())       
    
    # mark nan with -1, if encoding not necessary 
    if not encode:
        train[cat_features] = train[cat_features].replace(0, -1)
        test[cat_features] = test[cat_features].replace(0, -1)    
        
    ######## high cardinality
    
    if high_cardinality:

        hc_features = train[cat_features].columns[train[cat_features].apply(lambda x: len(x.value_counts())) > hc_treshold]
        target_mean = target.mean()
        S = {}

        for c in hc_features:

            if high_cardinality == "sr":
                # supervised ratio 
                group_means = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)], axis=1).groupby(c).mean()
                group_means = group_means.target.to_dict()
                for group in train[c].value_counts().index:
                    S[group] = group_means[group]

            if high_cardinality=="woe":
                # weight of evidence
                group_y1 = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)], axis=1).\
                groupby([c]).agg('sum')
                group_y0 = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)], axis=1).\
                groupby([c]).agg('count') - group_y1
                y1 = (target==1).sum()
                y0 = (target==0).sum()
                woe = np.log(((group_y1) / y1) / ((group_y0) / y0))
                for i,v in zip(woe.index, np.where(np.isinf(woe), 0, woe)):
                    S[i] = v[0]

            if high_cardinality=="smoothing":
                # empirical bayes (smoothing for small group)
                group_means = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)], axis=1).groupby(c).mean()
                group_means = group_means.target.to_dict()
                group_counts = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)], axis=1).groupby(c).agg('count')
                group_counts = group_counts.target.to_dict()

                def smoothing_function(n, k, f):
                    return 1 / (1 + np.exp(-(n-k)/f))

                for group in train[c].value_counts().index:
                    lam = smoothing_function(n=group_counts[group], k=eb_k, f=eb_f)
                    S[group] = lam*group_means[group] + (1-lam)*target_mean

            # transform train
            train[c+'_avg'] = train[c].apply(lambda x: S[x]).copy()

            # transform test
            def hc_transform_test(x):
                if x in S: 
                    return S[x]
                else:
                    return target_mean

            test[c+'_avg'] = test[c].apply(hc_transform_test).copy()

        # drop hc features 
        if hc_drop:
            train.drop(hc_features, axis=1, inplace=True)
            test.drop(hc_features, axis=1, inplace=True)

        # update cat features 
        cat_features = sorted(list(set(cat_features).difference(hc_features)))

    ######## for linear models 
#     print(train.shape)
#     if True:
#         return num_features, train
    
    # fill missings
    if fill_num in ['mean', 'median']:
        imputer = Imputer(strategy=fill_num)
        train[num_features] = imputer.fit_transform(train[num_features])
        test[num_features] = imputer.transform(test[num_features])
    elif fill_num < 0:
        train[num_features] = train[num_features].fillna(fill_num)
        test[num_features] = test[num_features].fillna(fill_num)
        
    # scaling
    if scaling=='standard':
        scaler = StandardScaler()
        train[num_features] = scaler.fit_transform(train[num_features])
        test[num_features] = scaler.transform(test[num_features])
    
    ######## encoding 
    if encode=='ohe':
        # one hot encoding, memory inefficient
        oh = OneHotEncoder(sparse=False)
        for c in cat_features:
            data=train[c].append(test[c])
            oh.fit(data.values.reshape(-1,1))            
            train_temp = oh.transform(train[c].values.reshape(-1,1))
            test_temp = oh.transform(test[c].values.reshape(-1,1))
            train = pd.concat([train, pd.DataFrame(train_temp, 
                                                   columns=[(c+"_"+str(i)) for i in data.value_counts().index],
                                                   index = train.index
                                                  )], axis=1)
            test = pd.concat([test, pd.DataFrame(test_temp, 
                                                 columns=[(c+"_"+str(i)) for i in data.value_counts().index],
                                                 index = test.index
                                                )], axis=1)
            # drop column
            train.drop(c, axis=1, inplace=True)
            test.drop(c, axis=1, inplace=True)
    
    if encode=='bin':
        # binary encoding 
        for c in cat_features:
            data=pd.DataFrame(train[c].append(test[c]), columns=[c])
            be = category_encoders.BinaryEncoder(cols=[c])
            be.fit(data)
            train_temp = be.transform(pd.DataFrame(train[c], columns=[c]))
            test_temp = be.transform(pd.DataFrame(test[c], columns=[c]))
            train = pd.concat([train, train_temp], axis=1)
            test = pd.concat([test, test_temp], axis=1)
            # drop column
            train.drop(c, axis=1, inplace=True)
            test.drop(c, axis=1, inplace=True)
       
    if keep:
        train = train.loc[:, keep]
        test = test.loc[:, keep]
        
    return train, test    


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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Imputer
import category_encoders
import gc

     
# load 

train, target = pd.read_csv('./input/orange_small_churn_data.train'), \
np.where(pd.read_csv('./input/orange_small_churn_labels.train', header=-1)==1, 1, 0).ravel()

opt_transform = depickle_it('opt_transform')

# model 

model = XGBClassifier(n_jobs=4, tree_method='gpu_hist', predictor = "cpu_predictor", objective="binary:logistic",
                      missing=-1, 
                      n_estimators=200, learning_rate=0.05,
                      max_depth=4, gamma=10, min_child_weight=2, reg_alpha=2, reg_lambda=1.3,
                      subsample=.8, colsample_bytree=.8,
                      scale_pos_weight=1
                     )

kf = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)

# hyperopt 

from hyperopt import hp
from hyperopt import fmin, tpe, Trials
import os 

if os.path.isfile('trials'):
    trials = depickle_it('trials')
else:
    trials = Trials()

space = {}
space['n_estimators'] = hp.randint('n_estimators', 20)*10 + 100
space['max_depth'] = hp.choice('max_depth', [3,4,5,6])
space['learning_rate'] = hp.uniform('learning_rate', 0.01, 0.1)
space['gamma'] = hp.randint('gamma', 5)*2
space['min_child_weight'] = hp.uniform('min_child_weight', 1.5, 2.5)
space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 0.5, 1.5)
space['subsample'] = hp.uniform('subsample', 0.7, 0.9)
space['colsample_bytree'] = hp.uniform('colsample_bytree', 0.7, 0.9)
space['reg_alpha'] = hp.uniform('reg_alpha', 1, 3)
space['reg_lambda'] = hp.uniform('reg_lambda', 1, 2)


def objective(params):

    i = len(trials.trials)
    print("*"*50+"\niter = {}".format(i))

    gc.collect()
    model.set_params(**params)
    
    score = np.mean(cross_val(train, target, model, kf, opt_transform, verbose=True))
    
    pickle_it(trials, 'trials')
    i += 1

    return -score

best = fmin(objective,
            space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials)

pickle_it(trials, 'trials')
pickle_it(best, 'best')
