import os
import pandas as pd

train = pd.read_csv(os.path.join('../practice/kaggle/titanic/data', 'train.csv'))
df = train

# counts
train.Survived.value_counts(normalize=True)

# group by
train['Survived'].groupby(train['Pclass']).mean()

# aggregate functions
df.Survived.groupby(df.Pclass).agg(['mean','count'])

# apply
train['Name_Len'] = train['Name'].apply(lambda x: len(x))

# cut
train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()

# fillna
train['Cabin'].fillna(0, inplace=True)

# recode
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'].mean()
train.loc[(df['Cabin'] != 0), 'Cabin'] = 1

