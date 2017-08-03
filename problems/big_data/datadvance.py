import numpy as np
import pandas as pd
import time


def make_data(M, N):
    df = pd.DataFrame({'feature1': np.random.randint(M, size=(N,)),
                       'feature2': np.random.randint(M, size=(N,)),
                       'time': np.random.rand(N)
                       })

    df.to_csv('incidents.csv', index_label='id')


M, N = 2, 10 ** 4
make_data(M, N)

# work data

df = pd.read_csv('incidents.csv')


def similar_incidents(df, dt):

    arr = df.values
    col_1, col_2, col_3 = arr[:,1], arr[:,2], arr[:,3]
    cache=[]

    def rowwise(row):
        count_sim = len(np.where((col_1 == row[1]) & (col_2 == row[2]) & (col_3 >= row[3] - dt) & (col_3 <= row[3])\
            [0])) - 1
        cache.append((row[0], count_sim))

    np.apply_along_axis(arr=arr, axis=1, func1d=rowwise)
    df = pd.DataFrame(cache, columns=['id','count'])
    df.to_csv('incidents_count.csv', index_label='id')

similar_incidents(df, 0.3)