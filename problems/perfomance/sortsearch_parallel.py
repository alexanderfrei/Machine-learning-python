from multiprocessing import Pool
from functools import partial
import time
import pandas as pd
import numpy as np


def make_data(M, N):

    df = pd.DataFrame({'feature1': np.random.randint(M, size=(N,)),
                       'feature2': np.random.randint(M, size=(N,)),
                       'time': np.random.rand(N)
                       })

    df.to_csv('incidents.csv', index_label='id')


def worker(arr, arr_full, dt):

    cache = []

    for row in arr:

        # tmp = arr_full[np.where(arr_full[:, 1] == row[1])] slow
        pos1 = np.searchsorted(arr_full[:, 1], row[1])
        pos2 = np.searchsorted(arr_full[:, 1], row[1]+1)
        tmp = arr_full[pos1:pos2, :]

        tmp2 = tmp[np.where(tmp[:, 2] == row[2])]
        tmp3 = tmp2[np.where(tmp2[:, 3] >= row[3] - dt)]
        count_sim = len(np.where(tmp3[:, 3] < row[3])[0])

        # count_sim = len(np.where((arr_full[:,1] == row[1]) & (arr_full[:,2] == row[2]) &
        #                          (arr_full[:,3] >= row[3] - dt) & (arr_full[:,3] < row[3]))[0])
        cache.append((row[0], count_sim))

    return cache


def count_similar(df, dt, n_jobs):

    start = time.time()
    num_of_processes = n_jobs

    arr_full = df.values
    args = np.array_split(arr_full, num_of_processes)
    arr_sorted = arr_full[arr_full.argsort(axis=0)[:, 1], :]

    pool = Pool(num_of_processes)
    result_list = pool.map(partial(worker, arr_full=arr_sorted, dt=dt), args)

    pool.close()
    pool.join()

    df = pd.DataFrame()
    for res in result_list:
        df = pd.concat([df, pd.DataFrame(res)])

    df.columns = ['id', 'count']
    df.to_csv('incidents_count.csv', index=False)

    print(time.time() - start)


if __name__ == "__main__":

    M, N, dt = 100, 10 ** 6, 0.3
    make_data(M, N)
    df = pd.read_csv('incidents.csv')

    count_similar(df, dt, n_jobs=4)
