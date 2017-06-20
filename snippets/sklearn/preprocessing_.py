# LabelEncoder with pandas df

df = pd.DataFrame(np.random.randn(5, 3),
                  columns=['a', 'b', 'c'])

df['d'] = ['a', 'b', 'c', 'a', 'b']
df['e'] = ['s', 'b', 'c', 'a', 'b']
df2 = df = df.loc[:, ['d', 'e']]

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

d = defaultdict(LabelEncoder)

df = df.loc[:, ['d', 'e']]
fit = df.apply(lambda x: d[x.name].fit_transform(x))  # Encoding the variable
fit.apply(lambda x: d[x.name].inverse_transform(x))  # Inverse the encoded
df2.apply(lambda x: d[x.name].fit_transform(x))  # use on new df


