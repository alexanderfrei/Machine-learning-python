
import pandas as pd
from multiprocessing import Pool
from functools import partial  # for additional worker parameters
import time
import numpy as np


def worker(arr, step):

    # do some unnecessary work
    return arr[::step]


def mapper(arr, step, n_jobs):

    pool = Pool(n_jobs)
    args = np.array_split(arr, n_jobs)  # main parameter

    result = pool.map(partial(worker, step=step), args) # list with len = n_jobs

    # run workers & merge results
    pool.close()
    pool.join()

    df = pd.DataFrame()
    for res in result:
        df = pd.concat([df, pd.DataFrame(res)])
    return df.shape

if __name__ == "__main__":

    start = time.time()

    step = 10
    arr = np.random.randint(1, 10, (10**5, 50))
    result = mapper(arr, step, n_jobs=4)

    print(result)
    print(time.time() - start)