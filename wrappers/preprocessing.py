# TODO: check and document functions

import pandas as pd
import numpy as np


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
