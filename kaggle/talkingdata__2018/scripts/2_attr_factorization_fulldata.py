
# coding: utf-8

import os 
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import gc 
import time

# os.getcwd()
os.chdir('D:\\z_old_projects\\talking_data\\scripts')


def minibatch_join(df, df_join, on, batch_size, out_cols):

    out = np.zeros((df.shape[0], len(out_cols)))
    batches = zip(range(0, df.shape[0], batch_size), 
                  list(range(batch_size, df.shape[0], batch_size)) + [df.shape[0]])
    
    for s,e in batches:
        print('batch {:d}-{:d}'.format(s, e))
        out[s:e, :] = df.iloc[s:e, :].merge(df_join, on=on, how='left')[out_cols]
    
    return pd.DataFrame(out, columns=out_cols)

# NCHUNK = 35000000
# OFFSET = 78000000
# nchunk=NCHUNK
# frm=nrows-OFFSET
# to=frm+nchunk

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

## DEBUG 
# to = 100000
# frm = 1
# train_df = pd.read_csv("../input/train.csv",
#                        skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, 
#                        usecols=['ip','app','device','os'])
# test_df = pd.read_csv("../input/test.csv", dtype=dtypes, usecols=['ip','app','device','os'], nrows=10000)

train_df = pd.read_csv("../input/train.csv", dtype=dtypes, 
                       usecols=['ip','app','device','os'])
test_df = pd.read_csv("../input/test.csv", dtype=dtypes, usecols=['ip','app','device','os'])

train_df = train_df.append(test_df)
del test_df
gc.collect()

print('df shape', train_df.shape)

# hashing users 
train_df['user'] = np.dot(train_df[['os', 'device', 'ip']].values, [10**8, 10**6, 1])
train_df.drop(['os', 'device'], 1, inplace=True)
tsvd = TruncatedSVD(3)

############### fac 1

print("factor 1: ip x app")
cross1 = np.log1p(pd.crosstab(train_df['ip'], train_df['app']))
factors1 = tsvd.fit_transform(cross1)

print("*"*2, "stack")
factors1 = pd.DataFrame(np.hstack([cross1.index.values.reshape(-1,1), factors1]), 
                        columns=['ip', 'fac1_1', 'fac1_2', 'fac1_3'])

print("*"*2, "join")
start = time.time()
out = minibatch_join(train_df, factors1, ['ip'], 10**6 * 10, ['fac1_1', 'fac1_2', 'fac1_3'])
print('join time: {:f}s'.format(time.time() - start))

out.to_feather('../dumps/factor1.feather')
del factors1, cross1, out
gc.collect()

############### fac 2 

print("factor 2: user x app")
user_counts = train_df['user'].value_counts()
mask = np.isin(train_df['user'].values, user_counts[user_counts > 10].index.values)
cross2 = np.log1p(pd.crosstab(train_df.loc[mask, 'user'], train_df.loc[mask, 'app']))
del user_counts, mask
gc.collect()
factors2 = tsvd.fit_transform(cross2)

print("*"*2, "stack")
factors2 = pd.DataFrame(np.hstack([cross2.index.values.reshape(-1,1), factors2]), 
                        columns=['user', 'fac2_1', 'fac2_2', 'fac2_3'])
del cross2
gc.collect()

print("*"*2, "join")
start = time.time()
out = minibatch_join(train_df, factors2, ['user'], 10**6 * 10, ['fac2_1', 'fac2_2', 'fac2_3'])
print('join time: {:f}s'.format(time.time() - start))

print('missings:')
print(out.isnull().sum(0))

out.to_feather('../dumps/factor2.feather')
