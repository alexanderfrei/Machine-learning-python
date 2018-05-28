import time
import pandas as pd
import numpy as np


def make_data(M, N):

    df = pd.DataFrame({'feature1': np.random.randint(M, size=(N,)),
                       'feature2': np.random.randint(M, size=(N,)),
                       'time': np.random.rand(N)
                       })

    df.to_csv('incidents.csv', index_label='id')


def count_similar(df, dt):

    start = time.time()

    # `hash` features
    df.iloc[:, 1] = df.iloc[:, 1] * 100 + df.iloc[:, 2]
    df.drop('feature2', inplace=True, axis=1)

    # sort df by features & time
    df.sort_values(axis=0, by=['feature1', 'time'], inplace=True)

    # vectors
    arr = df.values
    features, times = arr[:, 1], arr[:, 2]
    feat_values = np.unique(features)  # unique feature values
    fv_len = len(feat_values)

    # dictionary, key = unique feature value
    cache = {}

    def count_(row):

        feat, t = row[0], row[1]

        # slice times by current feature

        if feat in cache:
            tmp = cache[feat]

        else:

            # find feat index in feat_value list
            feat_index = np.where(feat_values == feat)[0]

            # search low index of feature by feature_index
            low_idx = np.searchsorted(features, feat_values[feat_index])[0]

            # search high index of feature by next feature value index & slice
            if feat_index < fv_len - 1:
                high_idx = np.searchsorted(features, feat_values[feat_index + 1])[0]
                tmp = times[low_idx:high_idx]
            else:
                tmp = times[low_idx:]

            # save in cache
            cache[feat] = tmp

        # count by delta
        if tmp.shape[0] > 1:
            return tmp.searchsorted(t + dt) - tmp.searchsorted(t - dt) - 1
        else:
            return 0

    out = np.hstack([arr,
                     np.apply_along_axis(arr=arr[:, 1:], axis=1, func1d=count_).reshape(-1, 1)])

    np.savetxt('out.csv', out[out[:, 0].argsort(), [[0], [3]]].T, delimiter=',', fmt="%i",
               header="id, count", comments='')

    print(time.time() - start)


if __name__ == "__main__":

    M, N, dt = 100, 10 ** 6, 0.3
    make_data(M, N)
    df = pd.read_csv('incidents.csv')

    count_similar(df, dt)
