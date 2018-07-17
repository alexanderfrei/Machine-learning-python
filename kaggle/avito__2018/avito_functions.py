import pandas as pd 
import numpy as np
# sklearn
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
# algo
import lightgbm as lg
# sparse
from scipy.sparse import hstack, csr_matrix 
from avito_classes import TargetEncoder

import pickle

def validation_split(X, y, ohe=None, n_valid=300000, n_test=100000, seed=10101):
    
    """
    X -- pandas df
    y -- numpy vector
    """

    print('run validation splitting..')
    n_train = X.shape[0] - (n_valid + n_test)
    
    # shuffling 
    idx = np.array(X.index)
    np.random.seed(seed)
    np.random.shuffle(idx)
    X, y = X.iloc[idx], y[idx]

    # splitting 
    x_train, x_valid, x_test = X[:n_train], X[n_train : n_train+n_valid], X[n_train+n_valid:]
    y_train, y_valid, y_test = y[:n_train], y[n_train : n_train+n_valid], y[n_train+n_valid:]
    if ohe is None:
        ohe_train, ohe_valid, ohe_holdout = None, None, None
    else:
        ohe_train, ohe_valid, ohe_holdout = ohe[:n_train], ohe[n_train : n_train+n_valid], ohe[n_train+n_valid:]            
    return x_train, x_valid, x_test, y_train, y_valid, y_test, ohe_train, ohe_valid, ohe_holdout


def preprocessing(df_train, df_test, map_dict, add_features=None):

    print('run preprocessing..')
    
    target = 'deal_probability'

    # get labels, merge 
    y = df_train[target].values.squeeze()
    X = df_train.drop([target], 1).append(df_test)
    X.index = np.arange(X.shape[0])

    # map additional information
    X['salaries'] = X.region.map(map_dict['salaries'])
    X['population'] = X.city.map(map_dict['population'])

    # merge additional features
    if not add_features is None:
        X = pd.concat([X, add_features], 1)
    
    # drop useless features 
    X = X.drop(['title', 'item_id', 'user_id'], 1)
   
    category_features = ['region', 'city', 
                         'parent_category_name', 'category_name', 
                         'param_1', 'param_2', 'param_3', 
                         'user_type', 'image_top_1']

    return X, y, category_features


def feature_engineering(X, category_features, factorize=False, price_bins=10):
    
    print('run feature engineering..')
    new_factors = []
    
    # numeric transformations 
    X['user_type_num'] = pd.factorize(X['user_type'])[0]
    X['price_log'] = np.log(X.price+0.0001)
    X['isn_log'] = np.log(X.item_seq_number+0.0001)
    X['price_log_div_salaries'] = X['price_log'] / X['salaries']  
    
    # bool
    X['price_exists'] = (~X.price.isnull()).astype(int)
    X['image_exists'] = (~X.image.isnull()).astype(int)
    X['descr_exists'] = (~X.description.isnull()).astype(int)
    
    X['population_groups'] = \
    ((X.population >= 1000000) * 1 + \
    ((X.population >= 500000) & (X.population < 1000000)) * 2 + \
    ((X.population >= 100000) & (X.population < 500000)) * 3 + \
    (X.population < 100000) * 4 - 1)     
    
    # date
    dt = pd.to_datetime(X.activation_date)
    X['weekday'] = dt.dt.weekday
    X['free_day'] = (dt.dt.dayofweek > 5).astype(int)  
    
    # groups, numeric 
    X = count_group_frac(X, 'price', ['region', 'category_name'])
    X = count_group_frac(X, 'price', ['region', 'param_1'])
    X = count_group_frac(X, 'price', ['region', 'param_2'])    
    X = count_group_frac(X, 'price', ['region', 'image_top_1'])    
    X = count_group_frac(X, 'price', ['city', 'category_name'])
    X = count_group_frac(X, 'price', ['city', 'param_1'])
    X = count_group_frac(X, 'price', ['city', 'param_2'])
    X = count_group_frac(X, 'price', ['city', 'image_top_1'])
    X = count_group_frac(X, 'price', ['image_top_1', 'category_name'])
    X = count_group_frac(X, 'price', ['image_top_1', 'param_1'])
    X = count_group_frac(X, 'price', ['image_top_1', 'param_2'])
    X = count_group_frac(X, 'price', ['population_groups', 'param_1'])
    
    # cutting
    X['price_log_cut'] = pd.cut(X['price_log'], bins=price_bins).cat.codes
    X['isn_log_cut'] = pd.cut(pd.Series(np.log(X.item_seq_number+0.0001)), 7).cat.codes
    
    # combine factors
    X, new_factors = combine_factors(X, 'price_log_cut', 'parent_category_name', new_factors)
    X, new_factors = combine_factors(X, 'price_log_cut', 'category_name', new_factors)
    X, new_factors = combine_factors(X, 'price_log_cut', 'region', new_factors)

    # features
    new_factors += ['price_exists', 'image_exists', 'descr_exists']
    new_factors += ['weekday', 'free_day']
    X.drop(['activation_date', 'image', 'description'], axis=1, inplace=True)
    #X.drop(['price_log'], axis=1, inplace=True) 
    
    category_features += new_factors
    category_features += ['population_groups']

    if factorize==True:
        for f in category_features:
            X[f] = pd.factorize(X[f])[0]
    if factorize=='pos':
        for f in category_features:
            X, _ = factorize_p(X, f)
    
    return X, category_features

##############################################################
#@ TRANSFORM CATEGORY @#
##############################################################


def factorize_p(df, f):
    """ positive factorization (>=0) """
    factor, levels = pd.factorize(df[f])
    factor += 1
    levels = [f + '_unk'] + [f + '_' + str(l) for l in levels]
    df[f] = factor
    return df, levels

def combine_factors(df, f1, f2, category_features):
    
    new_feature = f1+'_x_'+f2
    category_features += [new_feature]
    print('-- combine factors:', new_feature)
    
    df[new_feature] = df[f1].astype(str).str.cat(df[f2].astype(str), sep='_')
    return df, category_features


def target_encoding(X, y, X_test, group, X_holdout=None):
    print('-- target encoding:', group)
    te = TargetEncoder(group, k=60, f=10)
    te.fit_transform(X, y)
    te.transform(X_test)
    if not X_holdout is None:
        te.transform(X_holdout)
        return X, X_test, X_holdout
    else:
        return X, X_test
    
##############################################################
#@ ONEHOT @#
##############################################################
   
def onehot(x, encoder, mode, sparse, cat=None):
    """ onehot encoding matrix x """
    
    assert mode in {'fit', 'transform'}
    
    # split cat & numeric
    if cat is None:
        is_num = False
    else:
        is_num = True
        if sparse:
            x_num = csr_matrix(x.drop(cat, 1).values)
        else:
            x_num = x.drop(cat, 1).values
        x = x[cat].copy(deep=True)
    # remove negatives and nan
    x = x.replace(-1, 2**21)
    x = x.fillna(2**21)
    # fit / transform
    if mode=='fit': 
        x = encoder.fit_transform(x)
    elif mode=='transform': 
        x = encoder.transform(x)
    # merge and return
    if is_num:
        if sparse:
            return hstack([x_num, x]).tocsr(), encoder 
        else:
            return np.hstack([x_num, x]), encoder
    else:
        if sparse:
            return x.to_csr(), encoder
        else:
            return np.array(x), encoder
    
def cat_onehot(x_train, x_valid, x_holdout=None, cat=None, sparse=True):
    """ onehot encoding all feature matrices in pipeline """
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=sparse)
    x_train, encoder = onehot(x_train, encoder, 'fit', sparse, cat)
    x_valid, encoder = onehot(x_valid, encoder, 'transform', sparse, cat)
    if not x_holdout is None:
        x_holdout, encoder = onehot(x_holdout, encoder, 'transform', sparse, cat)
        return x_train, x_valid, x_holdout, encoder
    else:
        return x_train, x_valid, encoder
    
#### ONEHOT BATCH (SMALL RAM/BIG DATA)

def batcher(vec, chunksize=10000):
    """ batch generator for feature vec """
    for i in np.arange(0, vec.shape[0], chunksize):
        yield vec[i:i+chunksize]

def onehot_vec(X, f, dict_levels):
    print('-- onehot:', f)
    vec = X[f].values
    vec_csr = None
    batches = batcher(vec)
    n = len(dict_levels[f])
    for batch in batches:
        dense = np.eye(n)[batch]
        if vec_csr is None:
            vec_csr = csr_matrix(dense)
        else:
            vec_csr = vstack([vec_csr, dense]).tocsr()
    return vec_csr



##############################################################
#@ TRANSFORM NUMERIC @#
##############################################################

def num_fillna(x_train, x_valid, x_holdout=None, num=None, strategy='median'):
       
    if num is None:
        num = x_train.columns.tolist()
    
    imputer = Imputer(strategy=strategy)
    x_train[num] = imputer.fit_transform(x_train[num])
    x_valid[num] = imputer.transform(x_valid[num])
    if x_holdout is None:
        return x_train, x_valid, imputer
    else:
        x_holdout[num] = imputer.transform(x_holdout[num]) 
        return x_train, x_valid, x_holdout, imputer

def num_scaling(x_train, x_valid, x_holdout=None, num=None, mode='z', copy=True):
    
    if mode=='z':
        scaler = StandardScaler(copy=copy)
    elif mode=='norm':
        scaler = MinMaxScaler(feature_range=(0,1))
    else:
        pass
    
    if num is None:
        num = x_train.columns.tolist()
    
    # fit 
    x_train[num] = scaler.fit_transform(x_train[num])
    # transform valid 
    x_valid[num] = scaler.transform(x_valid[num])
    
    # transform holdout
    if x_holdout is None:
        return x_train, x_valid, scaler
    else:
        x_holdout[num] = scaler.transform(x_holdout[num])
        return x_train, x_valid, x_holdout, scaler


##############################################################
#@ COUNTERS @#
##############################################################

def count_group_frac(df, feature, group_levels):
    
    name = feature + '_x_' + '__'.join(group_levels) + '_frac'
    print('-- count fraction', name)
    
    group = df.groupby(group_levels)[feature].mean().reset_index()
    group_num = df[group_levels].merge(how='left', 
                                       on=group_levels, 
                                       right=group)[feature]
    
    df[name] = df[feature] / group_num
    return df

##############################################################
#@ METRICS @#
##############################################################

def rmse(y, yhat):
    return np.sqrt(np.mean(np.power(y.squeeze() - yhat.squeeze(), 2)))

##############################################################
#@ TRAIN @#
##############################################################

def lg_train(lg_params, xgtrain, xgvalid, num_boost_round, early_stopping_rounds, verbose_eval=10):
    evals_results = {}
    bst = lg.train(lg_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train', 'valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval)
    return bst, evals_results

def train_sklearn(algo, params, X, y):
    """ train sklearn model with holdout sample (train + valid + holdout) """
    model = algo(**params)
    model.fit(X[0], y[0])
    pred_val = model.predict(X[1])
    pred_holdout = model.predict(X[2])
    print(rmse(y[1], pred_val), rmse(y[2], pred_holdout))
    return pred_val, pred_holdout, model

def train_sklearn_valid(algo, params, X_train, X_test, y_train, y_test):
    """ train sklearn model without holdout sample (train + test) """
    model = algo(**params)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    print(rmse(y_test, pred_test))
    return pred_test, model

def oof_prediction(model, data, y, nfolds=4):
    """ 
    out-of-fold prediction 
    create additional prediction for training or blending
    input data -- list, X train + valid (+ holdout)
    output -- list, train + valid (+ holdout)
    """
    
    train_pred = np.zeros(data[0].shape[0])
    valid_pred = np.zeros(data[1].shape[0])
    errors = np.zeros(nfolds)
    if len(data) == 3: holdo_pred = np.zeros(data[2].shape[0])
    
    for i, (train_idx, test_idx) in enumerate( KFold(nfolds).split(data[0]) ):
        X_train, X_test = data[0][train_idx], data[0][test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        
        # predict 
        train_pred[test_idx] = model.predict(X_test)
        valid_pred += model.predict(data[1])
        if len(data) == 3: holdo_pred += model.predict(data[2])
        errors[i] = rmse(y_test, model.predict(X_test))
    
    print("{:.5f}+-{:.5f}".format(errors.mean(), errors.std()))
    
    valid_pred /= nfolds
    if len(data) == 3: 
        holdo_pred /= nfolds    
        return [train_pred, valid_pred, holdo_pred]
    else:
        return [train_pred, valid_pred]

##############################################################
#@ SAVE / LOAD @#
##############################################################

def load_fe(name):
    with open('../fe/' + name + '.pkl', 'rb') as file: fe_dict = pickle.load(file=file)
    return fe_dict

def save_pred_holdout(data, name):
    """ 
    save predictions for blending 
    output -- preds = dict with keys `valid`, `holdout`, `test` 
    input -- data list of predictions
    """
    assert isinstance(data, list)
    assert len(data) == 3
    
    preds['valid'] = data[0]
    preds['holdout'] = data[1]
    preds['test'] = data[2]

    with open('../blending/'+name+'.pkl', 'wb') as f: pickle.dump(obj=preds, file=f)
