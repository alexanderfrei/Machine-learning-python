import math
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

os.chdir("d:/datasets/sber")


# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]



# load
df_train = pd.read_csv("train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)
# df_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'])

id_test = df_test['id']


# indexing
df_fix = pd.read_csv("BAD_ADDRESS_FIX.csv", sep=",")
df_all = pd.concat([df_train, df_test])

fix_cols = df_fix.columns[1:].values
fix_ids = df_fix.iloc[:, 0]

df_all.set_index('id', inplace=True)
df_fix.set_index('id', inplace=True)



# fix
def replace_col_value_by_ids(col, df_fix):
    return df_fix.loc[:, col.name]

df_all.loc[fix_ids, fix_cols] = df_all.loc[fix_ids, fix_cols].apply(replace_col_value_by_ids, df_fix=df_fix)



# ylog_train_all = np.log1p(df_train['price_doc'].values)  # log(y + 1)
ylog_train_all = df_train['price_doc']
df_all.drop('price_doc', axis=1, inplace=True)  # drop id and y
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')


# Add month-year counts
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Relative squares and floor
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)


# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy array
X_all = df_values.values

# Create a validation set, with last 20% of data
num_train = len(df_train)
num_val = int(num_train * 0.2)

X_train_all = X_all[:num_train]
X_train = X_all[:num_train-num_val]
X_val = X_all[num_train-num_val:num_train]

ylog_train = ylog_train_all[:-num_val]
ylog_val = ylog_train_all[-num_val:]

X_test = X_all[num_train:]
df_columns = df_values.columns



# create xgboost objects
dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)
dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)
dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

# grid

eta = [0.05]
max_depth = [4, 5, 6]
subsample = [0.7]
colsample_bytree = [0.7, 0.9]
gamma = [0.0, 0.1, 0.3, 0.5]

from itertools import product
params_grid = product(eta, max_depth, subsample, colsample_bytree, gamma)

default_params = {
    'eta': 0.075,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'gamma': 0,
    'updater': 'grow_gpu'
}

watchlist  = [ (dtrain,'train'), (dval,'eval')]
def rmsle_eval(preds, dtrain):
    labels = dtrain.get_label()
    return 'rmsle', np.sqrt(np.mean(np.power(np.log1p(preds)-np.log1p(labels), 2)))

grid_result = []

for i, params in enumerate(params_grid):

    cache_params = default_params.copy()
    cache_params['eta'] = params[0]
    cache_params['max_depth'] = params[1]
    cache_params['subsample'] = params[2]
    cache_params['colsample_bytree'] = params[3]
    cache_params['gamma'] = params[4]
    cache_model = xgb.train(cache_params, dtrain, 2000, watchlist, feval=rmsle_eval,
                            verbose_eval=50, early_stopping_rounds=20)
    grid_result.append((params, cache_model.best_score))

print(sorted(grid_result, key= lambda x: x[1])[:5])


# fit num boost rounds for best model

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'objective': 'reg:linear',
    'updater': 'grow_gpu'
}


partial_model = xgb.train(xgb_params, dtrain, 1000,  watchlist, feval=rmsle_eval,
                       early_stopping_rounds=20, verbose_eval=50)


num_boost_round = partial_model.best_iteration  # assign best number of iterations

model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)  # fit model on all train data
fig, ax = plt.subplots(1, 1, figsize=(8, 16))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)


# submission
ylog_pred = model.predict(dtest)
y_pred = np.exp(ylog_pred) - 1

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

hash = ""
if os.path.exists('sub.csv'): hash += str(np.random.randint(0, 10 ** 5 ) * np.random.randint(0, 10 ** 3 ) )
df_sub.to_csv('sub{}.csv'.format(hash), index=False, )


