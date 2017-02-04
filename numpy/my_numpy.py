import numpy as np


def cross_freq(arr):
    shape = arr.shape[1]
    f = np.zeros((shape, shape))
    for j in range(0, shape):
        for i in range(j, shape):
            ind = np.where(np.logical_and(~np.isnan(arr[..., j]), ~np.isnan(arr[..., i])))
            f[j, i], f[i, j] = ind[0].shape[0], ind[0].shape[0]
    return f

