import os
import pandas as pd

train = pd.read_csv(os.path.join('../practice/kaggle/titanic/data', 'train.csv'))
df = train

# window width option
pd.set_option('display.width', 1500)

# explore
df.info()
df.describe()
df.head()
df.tail()
df.nlargest(5, ['Pclass', 'Age'])
df.nsmallest(5, ['Pclass', 'Age'])

# counts
train.Survived.value_counts(normalize=True)

# group by
train['Survived'].groupby(train['Pclass']).mean()

# aggregate wrappers
df.Survived.groupby(df.Pclass).agg(['mean','count'])

# apply
train['Name_Len'] = train['Name'].apply(lambda x: len(x))
# value_counts by all columns
df.apply(pd.Series.value_counts)

# cut
pd.cut(train['Age'],3)

# fillna
train['Cabin'].fillna(0, inplace=True)

# recode
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'].mean()
train.loc[(df['Cabin'] != 0), 'Cabin'] = 1

# one hot
pd.get_dummies(pd.cut(train['Pclass'],5), prefix="Pclass")

# sorting
df.sort_values(by=('Age'))

# change columns name
df.columns = [x.lower() for x in df.columns]

