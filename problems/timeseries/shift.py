import numpy as np
import pandas as pd
import statsmodels.api as sm

# load data with absence rows

data = sm.datasets.co2.load_pandas()
df = data.data
df = df['co2'].resample('MS').mean()
df = df.fillna(df.bfill())
df = pd.DataFrame(df)

df = df['1959']
df['rnd'] = np.random.randint(0, 1000, df.shape[0])
df['id'] = np.arange(0, df.shape[0])

drop_indices = np.random.choice(df.index, 4, replace=False)
df = df.drop(drop_indices)

# create 2 copy of variables with shift by 1 and 2 months

# 1: fill df  with missing months - reindex
# 2: shift 2 times

# 1
date_index = pd.date_range('1959/1/1', periods=12, freq='MS')
df = df.reindex(index=date_index)
drop_indexes = df[df.sum(axis=1) == 0].index

# 2
df1 = df.shift(1)
df2 = df.shift(2)

# rename cols
colnames = df.columns.values

cols1 = {i: i+"_1" for i in colnames}
df1.rename(columns=cols1, inplace=True)
cols2 = {i: i+"_2" for i in colnames}
df2.rename(columns=cols2, inplace=True)

# merge
df = pd.concat([df, df1, df2], axis=1)
df.drop(drop_indexes, inplace=True)

df