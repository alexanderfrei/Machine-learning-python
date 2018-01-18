# TODO: check and document functions

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Imputer
import category_encoders

def pd_to_np(df):
    """ pandas DF to numpy ARRAY
    :param df: pandas dataframe
    :return: numpy array
    """
    arr_ip = [tuple(i) for i in df.as_matrix()]
    dt = np.dtype([(str(i), j) for i,j in zip(df.dtypes.index, df.dtypes)])
    arr = np.array(arr_ip, dtype=dt)
    return arr


def recode(df, column, bins):
    pd.get_dummies(pd.cut(df[column],bins), prefix=column)


def dummies(train, test, columns):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column + '_' + i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix=column)[good_cols]), axis=1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix=column)[good_cols]), axis=1)
        del train[column]
        del test[column]
    return train, test


def cross_freq(arr):
    shape = arr.shape[1]
    f = np.zeros((shape, shape))
    for j in range(0, shape):
        for i in range(j, shape):
            ind = np.where(np.logical_and(~np.isnan(arr[..., j]), ~np.isnan(arr[..., i])))
            f[j, i], f[i, j] = ind[0].shape[0], ind[0].shape[0]
    return f


def transform(train, test, target, to_drop=None,
              high_cardinality="smoothing", hc_treshold=10, hc_drop=None,  # high cardinality type
              eb_k=50, eb_f=10,  # parameters for high cardinality smoothing function
              encode=None,  # categorical
              fill_num=-1, scaling=None,  # continuous
              feature_names=None, feature_dtypes=None, keep=None
              ):
    """ 
    data preprocessing 

    :train, test: pandas DataFrame/ numpy array
    :high_cardinality: way to handle categorical features with high number of levels ["sr", "woe", "smoothing"]
    :hc_drop: drop or keep high cardinality features 
    :encode: category encoding, 'ohe' = one hot, 'bin' = binary
    :fill_num: fill nan for continuous features, -1 = with -1, ['mean', 'median'] = strategy for Imputer
    :scaling: 'standard' = StandartScaler
    :feature_names: list, columns for numpy array
    :feature_dtypes: pd.Series, dtypes from dataframe
    :keep: list of columns to keep

    category features should have type 'object'
    """

    # checks

    feature_names = list(feature_names)
    if (isinstance(train, np.ndarray) or isinstance(test, np.ndarray)) and not feature_names:
        print("Define feature names for numpy array!")
        return 0
        # raise FeatureNamesError("Feature names had not defined for np.ndarray")

    # np to pd
    if isinstance(train, np.ndarray):
        train = pd.DataFrame(train, columns=feature_names)
        dt = feature_dtypes.to_dict()
        for col in train.columns:
            train[col] = train[col].astype(dt[col])
    if isinstance(test, np.ndarray):
        test = pd.DataFrame(test, columns=feature_names)
        dt = feature_dtypes.to_dict()
        for col in train.columns:
            test[col] = test[col].astype(dt[col])

    # remove duplicates
    if to_drop:
        train = train.drop(to_drop, axis=1)
        test = test.drop(to_drop, axis=1)

    ######## categorical features

    cat_features = train.columns[train.dtypes == 'object']
    num_features = train.columns[train.dtypes != 'object']

    # factorize
    le = LabelEncoder()
    train[cat_features] = train[cat_features].fillna('-1')
    test[cat_features] = test[cat_features].fillna('-1')
    for c in cat_features:
        data = train[c].append(test[c])
        le.fit(data.values.tolist())  # nan = 0 level
        train[c] = le.transform(train[c].values.tolist())
        test[c] = le.transform(test[c].values.tolist())

        # mark nan with -1, if encoding not necessary
    if not encode:
        train[cat_features] = train[cat_features].replace(0, -1)
        test[cat_features] = test[cat_features].replace(0, -1)

        ######## high cardinality

    if high_cardinality:

        hc_features = train[cat_features].columns[
            train[cat_features].apply(lambda x: len(x.value_counts())) > hc_treshold]
        target_mean = target.mean()
        S = {}

        for c in hc_features:

            if high_cardinality == "sr":
                # supervised ratio
                group_means = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)],
                                        axis=1).groupby(c).mean()
                group_means = group_means.target.to_dict()
                for group in train[c].value_counts().index:
                    S[group] = group_means[group]

            if high_cardinality == "woe":
                # weight of evidence
                group_y1 = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)], axis=1). \
                    groupby([c]).agg('sum')
                group_y0 = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)], axis=1). \
                               groupby([c]).agg('count') - group_y1
                y1 = (target == 1).sum()
                y0 = (target == 0).sum()
                woe = np.log(((group_y1) / y1) / ((group_y0) / y0))
                for i, v in zip(woe.index, np.where(np.isinf(woe), 0, woe)):
                    S[i] = v[0]

            if high_cardinality == "smoothing":
                # empirical bayes (smoothing for small group)
                group_means = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)],
                                        axis=1).groupby(c).mean()
                group_means = group_means.target.to_dict()
                group_counts = pd.concat([train[c], pd.DataFrame(target, columns=['target'], index=train.index)],
                                         axis=1).groupby(c).agg('count')
                group_counts = group_counts.target.to_dict()

                def smoothing_function(n, k, f):
                    return 1 / (1 + np.exp(-(n - k) / f))

                for group in train[c].value_counts().index:
                    lam = smoothing_function(n=group_counts[group], k=eb_k, f=eb_f)
                    S[group] = lam * group_means[group] + (1 - lam) * target_mean

            # transform train
            train[c + '_avg'] = train[c].apply(lambda x: S[x]).copy()

            # transform test
            def hc_transform_test(x):
                if x in S:
                    return S[x]
                else:
                    return target_mean

            test[c + '_avg'] = test[c].apply(hc_transform_test).copy()

        # drop hc features
        if hc_drop:
            train.drop(hc_features, axis=1, inplace=True)
            test.drop(hc_features, axis=1, inplace=True)

        # update cat features
        cat_features = sorted(list(set(cat_features).difference(hc_features)))

    ######## test
    #     print(train.shape)
    #     if True:
    #         return num_features, train

    ######## for linear models

    # fill missings
    if fill_num in ['mean', 'median']:
        imputer = Imputer(strategy=fill_num)
        train[num_features] = imputer.fit_transform(train[num_features])
        test[num_features] = imputer.transform(test[num_features])
    elif fill_num < 0:
        train[num_features] = train[num_features].fillna(fill_num)
        test[num_features] = test[num_features].fillna(fill_num)

    # scaling
    if scaling == 'standard':
        scaler = StandardScaler()
        train[num_features] = scaler.fit_transform(train[num_features])
        test[num_features] = scaler.transform(test[num_features])

    ######## encoding
    if encode == 'ohe':
        # one hot encoding, memory inefficient
        oh = OneHotEncoder(sparse=False)
        for c in cat_features:
            data = train[c].append(test[c])
            oh.fit(data.values.reshape(-1, 1))
            train_temp = oh.transform(train[c].values.reshape(-1, 1))
            test_temp = oh.transform(test[c].values.reshape(-1, 1))
            train = pd.concat([train, pd.DataFrame(train_temp,
                                                   columns=[(c + "_" + str(i)) for i in data.value_counts().index],
                                                   index=train.index
                                                   )], axis=1)
            test = pd.concat([test, pd.DataFrame(test_temp,
                                                 columns=[(c + "_" + str(i)) for i in data.value_counts().index],
                                                 index=test.index
                                                 )], axis=1)
            # drop column
            train.drop(c, axis=1, inplace=True)
            test.drop(c, axis=1, inplace=True)

    if encode == 'bin':
        # binary encoding
        for c in cat_features:
            data = pd.DataFrame(train[c].append(test[c]), columns=[c])
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


def cut(train, test, var, n_bins=10):
    """ 
    cut - split continuous by bins 
    
    train: pandas dataframe
    test: pandas dataframe
    var: var name, string
    return: train, test tuple
    
    """

    train.loc[:, var + 'qcut'] = pd.DataFrame(np.zeros((train.shape[0], 1)))
    test.loc[:, var + 'qcut'] = pd.DataFrame(np.zeros((test.shape[0], 1)))

    if (train.loc[:, var] == -1).sum() > 0:

        mask = train.loc[:, var] != -1
        sr, bins = pd.qcut(train.loc[mask, var], q=n_bins, retbins=True, duplicates='drop')
        train.loc[mask, var + 'qcut'] = sr
        train.loc[~mask, var + 'qcut'] = '-1'
        train.loc[:, var + 'qcut'] = train.loc[:, var + 'qcut'].astype('category')

        mask = test.loc[:, var] != -1
        test.loc[mask, var + 'qcut'] = pd.cut(test.loc[mask, var], bins=bins)
        test.loc[~mask, var + 'qcut'] = '-1'
        test.loc[:, var + 'qcut'] = test.loc[:, var + 'qcut'].astype('category')

    else:

        sr, bins = pd.qcut(train.loc[:, var], q=n_bins, retbins=True, duplicates='drop')
        train.loc[:, var + 'qcut'] = sr
        test.loc[:, var + 'qcut'] = pd.cut(test.loc[:, var], bins=bins)

    new_index = []
    for i in range(len(train.loc[:, var + 'qcut'].cat.categories)):
        new_index.append(str(train.loc[:, var + 'qcut'].cat.categories[i]))

    train.loc[:, var + 'qcut'].cat.categories = new_index
    test.loc[:, var + 'qcut'].cat.categories = train.loc[:, var + 'qcut'].cat.categories

    train.loc[:, var + 'qcut'] = train.loc[:, var + 'qcut'].astype(str)
    test.loc[:, var + 'qcut'] = test.loc[:, var + 'qcut'].astype(str)

    lbl = LabelEncoder()
    lbl.fit(list(train[var + 'qcut'].values) + list(test[var + 'qcut'].values))
    train[var + 'qcut'] = lbl.transform(list(train[var + 'qcut'].values))
    test[var + 'qcut'] = lbl.transform(list(test[var + 'qcut'].values))

    return train, test