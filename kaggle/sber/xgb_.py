def main():

    import math
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import matplotlib.pyplot as plt
    import os

    os.chdir("d:/datasets/sber")

    # load data
    df_train = pd.read_csv("train.csv", parse_dates=['timestamp'])
    df_test = pd.read_csv("test.csv", parse_dates=['timestamp'])
    df_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'])
    df_fix = pd.read_csv("BAD_ADDRESS_FIX.csv", sep=",")

    id_test = df_test['id']
    id_train = df_train['id']


    # fix
    df_all = pd.concat([df_train, df_test])

    fix_cols = df_fix.columns[1:].values
    fix_ids = df_fix.iloc[:, 0]

    df_all.set_index('id', inplace=True)
    df_fix.set_index('id', inplace=True)

    def replace_col_value_by_ids(col, df_fix):
        return df_fix.loc[:, col.name]

    df_all.loc[fix_ids, fix_cols] = df_all.loc[fix_ids, fix_cols].apply(replace_col_value_by_ids, df_fix=df_fix)


    num_train = len(df_train)
    y = np.log1p(df_train['price_doc'].values)  # log(y + 1)
    df_all.drop('price_doc', axis=1, inplace=True)
    df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')


    # feature engineering

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

    # macro feature selection VIF
    # From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
    macro_vif = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
    "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
    "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

    s = set(df_macro.columns)
    macro_columns = list(s.difference(macro_vif))
    df_all.drop(macro_columns, axis=1, inplace=True)

    # df_all.drop(['timestamp'], axis=1, inplace=True)

    # Deal with categorical values
    df_numeric = df_all.select_dtypes(exclude=['object'])
    df_obj = df_all.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    df_values = pd.concat([df_numeric, df_obj], axis=1)
    df_columns = df_values.columns
    X_all = df_values.values


    # train/validation split

    num_val = int(num_train * 0.2)

    X_train_all = X_all[:num_train]
    X_train = X_all[:num_train-num_val]
    X_val = X_all[num_train-num_val:num_train]
    y_train = y[:-num_val]
    y_val = y[-num_val:]

    X_test = X_all[num_train:]


    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    default_params = {
        'learning_rate': 0.075,
        'max_depth': 5,
        'subsample': 1.0,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'n_estimators': 1000
    }

    xgb_model = XGBRegressor(**default_params)

    fit_params = {
        'eval_set': [(X_val, y_val)],
        'early_stopping_rounds': 20,
        'verbose': 50
    }

    params = dict(learning_rate = [0.03, 0.05],
                  max_depth = [5, 6],
                  subsample = [0.9],
                  colsample_bytree = [0.9],
                  gamma = [0, 0.5])

    grid_search = GridSearchCV(xgb_model, param_grid=params, cv=5, n_jobs=3, fit_params=fit_params)
    grid_search.fit(X_train, y_train)

    return grid_search

if __name__ == "__main__":
    dict = main()
    with open('grid.txt', 'w') as f:
        for k, v in dict.cv_results_.items():
            f.writelines("{}: {}\n".format(k, v))

