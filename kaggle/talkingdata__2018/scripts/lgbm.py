import pickle
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
from collections import deque


def lgb_fulltrain(lgb_params, xgtrain, num_boost_round, lr_decay, nl_random, md_random, verbose_eval=10):
    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     num_boost_round=num_boost_round,
                     verbose_eval=verbose_eval, 
                     callbacks= [lgb.reset_parameter(learning_rate = lr_decay, 
                                                     num_leaves = nl_random,
                                                     max_depth = md_random)]
                     )
    return bst

def lgb_train(lgb_params, xgtrain, xgvalid, num_boost_round, early_stopping_rounds, lr_decay, nl_random, md_random, verbose_eval=10):
    evals_results = {}
    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval, 
                     callbacks= [lgb.reset_parameter(learning_rate = lr_decay, 
                                                     num_leaves = nl_random,
                                                     max_depth = md_random)],
                     feval=None)
    return bst, evals_results

np.random.seed(42)

early_stopping_rounds=30
num_boost_round=330

PREDICTORS = ['app', 'channel', 'os', 'device', 'hour', 'nextClick'] + ['X'+str(i) for i in [0,1,2,3,4,5,7,8,9,11,13,14,15]] + \
             ['fac1_1', 'fac1_2', 'fac1_3', 'fac2_1', 'fac2_2', 'fac2_3']
categorical = ['app', 'device', 'os', 'channel', 'hour']

lr_decay = [0.1] * num_boost_round
nl_random = [31] * num_boost_round
md_random = [6] * num_boost_round

# lr_decay = [0.25] * 100 + [0.2] * 100 + [0.15] * 100 + [0.1] * 700
# nl_random = np.random.choice([7, 15, 31], 1000).tolist()
# md_random = [7] * 1000
# for s,i in enumerate(nl_random):
#     np.random.seed(s+1)
#     md = int(np.log2(i+1))
#     md_random.append(np.random.choice([md+1, md+2, md+3], 1))
# np.random.seed(42)

dump = 0
from_dump = 1
dump_name = ''
full_train = 1
predict = 1
DEBUG = 0
feature_eliminate = 0

FILENO = 12

NCHUNK = 184903890
OFFSET = 184903890
val_size= 50000000

# NCHUNK = 35000000
# OFFSET = 78000000
# val_size= 5000000

MISSING32 = 999999999
MISSING8 = 255

inpath = '../input/'
outpath = '../sub'
cores = 4

def rolling_counts(df, hash_col, window):
    print('rolling ', hash_col, str(window), '...')

    output = [0]
    data, inttime = df[hash_col].values, df['inttime'].values
    values, counts = deque(maxlen=10000000), {}
    values.append(data[0])
    counts[data[0]] = 1
    start_time = inttime[0]
    start_index = 0

    for val, cur_time in zip(data[1:], inttime[1:]):

        # recursive remove by time delta
        if cur_time - start_time > window:
            for _ in range(len(values)):
                if cur_time - start_time <= window: 
                    break
                start_index += 1
                start_time = inttime[start_index]
                counts[values.popleft()] -= 1

        # add
        output.append(counts.get(val, 0))
        values.append(val)
        counts[val] = counts.get(val, 0) + 1
    
    df['_'.join(['roll', hash_col, str(window)])] = output
    return df

def to_hash(df, cols, hashname):
    # hash for dummies
    hash_mult = [100000000, 10000000, 100000, 1]
    if len(cols) == 4:
        df[hashname] = df[cols[0]] * hash_mult[0] + \
                       df[cols[1]] * hash_mult[1] + \
                       df[cols[2]] * hash_mult[2] + \
                       df[cols[3]] * hash_mult[3]
    elif len(cols) == 3:
        df[hashname] = df[cols[0]] * hash_mult[0] + \
                       df[cols[1]] * hash_mult[1] + \
                       df[cols[2]] * hash_mult[3]
    elif len(cols) == 2:
        df[hashname] = df[cols[0]] * hash_mult[0] + \
                       df[cols[1]] * hash_mult[3]
    df[hashname] = pd.factorize(df[hashname])[0]
    return df   

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

target = 'is_attributed'
fileno = FILENO
debug = DEBUG


if not from_dump:

    if debug:
        print('*** debug parameter set: this is a test run for debugging purposes ***')
    
    nrows=184903890
    nchunk=NCHUNK
    frm=nrows-OFFSET
    if debug:
        frm=0
        nchunk=100000
        val_size=10000
    to=frm+nchunk

    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv(inpath+"train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    print('loading test data...')
    if debug:
        test_df = pd.read_csv(inpath+"test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv(inpath+"test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    train_df['click_id'] = MISSING32
    train_df['click_id'] = train_df.click_id.astype('uint32')


    len_train = len(train_df)
    test_df['is_attributed'] = MISSING8
    test_df['is_attributed'] = test_df.is_attributed.astype('uint8')
    train_df=train_df.append(test_df)

    del test_df
    gc.collect()

    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    # train_df['inttime'] = train_df.click_time.astype(np.int64) // 10**9

    print('Extracting aggregation features...')
    if 'X0' in PREDICTORS: train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8'); gc.collect()
    if 'X1' in PREDICTORS: train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1'); gc.collect()
    if 'X2' in PREDICTORS: train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8'); gc.collect()
    if 'X3' in PREDICTORS: train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint8'); gc.collect()
    if 'X4' in PREDICTORS: train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8'); gc.collect()
    if 'X5' in PREDICTORS: train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16'); gc.collect()
    if 'X6' in PREDICTORS: train_df = do_countuniq( train_df, ['app'], 'channel', 'X6'); gc.collect()
    if 'X7' in PREDICTORS: train_df = do_cumcount( train_df, ['ip'], 'os', 'X7'); gc.collect()
    if 'X8' in PREDICTORS: train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8'); gc.collect()
    if 'X9' in PREDICTORS: train_df = do_count( train_df, ['ip', 'day', 'hour'], 'X9'); gc.collect(); train_df['X9'] = train_df['X9'].astype('uint16')
    if 'X10' in PREDICTORS: train_df = do_count( train_df, ['ip', 'app'], 'X10'); gc.collect(); train_df['X10'] = train_df['X10'].astype('uint16')
    if 'X11' in PREDICTORS: train_df = do_count( train_df, ['ip', 'app', 'os'], 'X11', 'uint16'); gc.collect(); train_df['X11'] = train_df['X11'].astype('uint16')
    if 'X12' in PREDICTORS: train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'X12'); gc.collect()
    if 'X13' in PREDICTORS: train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'X13'); gc.collect()
    if 'X14' in PREDICTORS: train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'X14'); gc.collect()
    if 'X15' in PREDICTORS: train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'X15'); gc.collect()

    # rolling counts
    # train_df = to_hash(train_df, ['app', 'device', 'os', 'ip'], 'roll1')
    # train_df = rolling_counts(train_df, 'roll1', 120)

    print('Doing nextClick...')

    def next_click(df):
        df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
        df['nextClick'] = (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time).astype(np.float32)
        # df['nextClick_shift'] = df['nextClick'].shift(1).values
    next_click(train_df)

    print("vars and data type: ")
    train_df.info()

    # drop 
    # train_df.drop(['inttime'], 1, inplace=True)

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    if dump:
        print("Dumping...")
        train_df.reset_index().to_feather('../dumps/kaggle_train' + dump_name + '.feather')
        val_df.reset_index().to_feather('../dumps/kaggle_val' + dump_name + '.feather')
        test_df.reset_index().to_feather('../dumps/kaggle_test' + dump_name + '.feather')
else:
    print('Load dumps...')
    train_df = pd.read_feather('../dumps/kaggle_train' + dump_name + '.feather').set_index('index')

    val_df = pd.read_feather('../dumps/kaggle_val' + dump_name + '.feather').set_index('index')
    # test_df = pd.read_feather('../dumps/kaggle_test' + dump_name + '.feather').set_index('index')
    # print(test_df.shape)

    ntrain, nfull = train_df.shape[0], train_df.shape[0] + val_df.shape[0]    

    # add factors 
    factor1 = pd.read_feather('../dumps/factor1' + dump_name + '.feather')
    factor2 = pd.read_feather('../dumps/factor2' + dump_name + '.feather')    
    factors = pd.concat([factor1, factor2], axis=1)
    del factor1, factor2
    gc.collect()
    train_df = pd.concat([train_df, factors[:ntrain]], axis=1)
    val_df = pd.concat([val_df, factors[ntrain:nfull]], axis=1)
    factors_test = factors[nfull:]
    del factors
    gc.collect()

    print(train_df.shape)
    print(val_df.shape)


predictors = PREDICTORS

# print(predictors)
# print(train_df.apply(lambda x: len(x.value_counts())))

print("Training...")
start_time = time.time()

objective='binary' 
metrics='auc'
verbose_eval=True 
categorical_features=categorical

lgb_params = {
    # 'device': 'gpu',
    # 'gpu_platform_id': 0,
    # 'gpu_device_id': 0,
    # 'gpu_use_dp': False,
    'boosting_type': 'gbdt',
    'objective': objective,
    'metric':metrics,
    'learning_rate': 0.1,
    'num_leaves': 31,  # 2^max_depth - 1
    'max_depth': 6,  # -1 means no limit
    'min_child_samples': 200,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.8,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 30,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 400, # because training data is extremely unbalanced 
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0.1,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': cores,
    'verbose': -1,
    'metric':metrics
}

if not full_train:
    
    print('Make xgtrain...')
    y_train = train_df[target].values
    xgtrain = lgb.Dataset(train_df[predictors].values.astype(np.float32), label=y_train,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
                          
    if not feature_eliminate: 
        del train_df
        gc.collect()


    print('Make xgval...')
    y_val = val_df[target].values
    xgvalid = lgb.Dataset(val_df[predictors].values, label=y_val,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    if not feature_eliminate: 
        del val_df
        gc.collect()


    bst, evals_results = lgb_train(lgb_params, xgtrain, xgvalid, num_boost_round, early_stopping_rounds, lr_decay, nl_random, md_random)
    print("\nModel Report")
    print("bst.best_iteration: ", bst.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst.best_iteration-1])    

else:

    train_df = pd.concat([train_df, val_df])
    y_train = train_df[target].values

    del val_df
    gc.collect()

    xgtrain = lgb.Dataset(train_df[predictors].values.astype(np.float32), label=y_train,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
                         
    bst = lgb_fulltrain(lgb_params, xgtrain, num_boost_round, lr_decay, nl_random, md_random)

print('[{}]: model training time'.format(time.time() - start_time))
with open('../dumps/model'+str(fileno), 'wb') as f: pickle.dump(bst, f)

if feature_eliminate:
    print("Feature eliminating...")

    if os.path.exists('../dumps/elim_dump.pkl'): 
        with open('../dumps/elim_dump.pkl', 'rb') as f: 
            scored = pickle.load(f)
    else:
        scored = []      
    
    to_eliminate = []
    predictors.sort()   
    best_score = evals_results['valid'][metrics][bst.best_iteration-1]
    improved = True

    while improved:

        improved = False
        for i, p in enumerate(predictors):
            print('eliminated', [p] + to_eliminate)
            
            cur_predictors = predictors[:i] + predictors[i+1:]
            if p in categorical: 
                ic = categorical.index(p)
                cur_cat = categorical[:ic] + categorical[ic+1:]
            else:
                cur_cat = categorical

            xgtrain = lgb.Dataset(train_df[cur_predictors].values.astype(np.float32), label=y_train,
                                  feature_name=cur_predictors,
                                  categorical_feature=cur_cat
                                  )        

            xgvalid = lgb.Dataset(val_df[cur_predictors].values, label=y_val,
                                  feature_name=cur_predictors,
                                  categorical_feature=cur_cat
                                  )        

            model, evals_results = lgb_train(lgb_params, xgtrain, xgvalid, num_boost_round, early_stopping_rounds, lr_decay, nl_random, md_random, 0)
            score = evals_results['valid'][metrics][model.best_iteration-1]

            if score > best_score: 
                best_score = score 
                improved = True
                remove = p

            scored.append((cur_predictors, score))
            with open('../dumps/elim_dump.pkl', 'wb') as f: pickle.dump(scored, f)

        if improved:
            to_eliminate.append(p)
            i = predictors.index(p)
            predictors = predictors[:i] + predictors[i+1:]
            if p in categorical: 
                i = categorical.index(p)
                categorical = categorical[:i] + categorical[i+1:]
            print(predictors)
            print('best score: {:.6f}'.format(score))

    print(predictors)
    print(categorical)
    print(best_score)
    with open('../dumps/best_predictors.pkl', 'wb') as f: pickle.dump(predictors, f)
    with open('../dumps/best_cat.pkl', 'wb') as f: pickle.dump(categorical, f)

if predict:

    test_df = pd.read_feather('../dumps/kaggle_test' + dump_name + '.feather').set_index('index')
    if from_dump: test_df = pd.concat([test_df, factors_test], axis=1)    
    print(test_df.shape)

    print("Predicting...")
    sub = pd.DataFrame()
    y_pred = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)
    outsuf = ''
    sub['click_id'] = test_df['click_id'].astype('uint32').values
    sub['is_attributed'] = y_pred
    if not debug:
        print("\nwriting...")
        sub.to_csv('../sub/sub_it%d'%(fileno)+'.csv', index=False, float_format='%.9f')
    print( sub.head(10) )

    print('Plot feature importances...')
    ax = lgb.plot_importance(bst, max_num_features=100)
    plt.savefig('FE.jpg')
    plt.show()    
