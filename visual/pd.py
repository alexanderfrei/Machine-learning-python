import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

##############################################################################################################
# histogram

# df = pd.DataFrame({'col1':np.random.randn(100),'col2':np.random.randn(100)})
# df.hist(layout=(1,2))
#
df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

# df4.diff().hist(alpha=0.5, color='k')
# df4.plot.hist(alpha=0.5)
# df4.plot.hist(stacked=True, bins=20)
# df4.a.plot.hist(orientation='horizontal', cumulative=True)

# # by group
# data = pd.Series(np.random.randn(1000))
# group_var = np.random.randint(0, 3, 1000)
# data.hist(by=group_var, alpha=0.9, bins=15, figsize=(12,8))

# plt.plot([ax1,ax2])
# print(ax1, ax2)

##############################################################################################################
# facets
# fig, axes = plt.subplots(ncols=2, figsize=(10,5))
# df4.plot.hist(ax=axes[0], stacked=True, bins=20)
# df4.plot.kde(ax=axes[1], stacked=True)

##############################################################################################################
# line plot

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
# ts.plot();
#
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
# df.plot();
#
# df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
# df3['A'] = pd.Series(list(range(len(df))))
# df3.plot(x='A', y='B');

##############################################################################################################
# bar plot

# df.ix[5].plot(kind='bar'); plt.axhline(0, color='k')

df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
# df2.plot.bar()

# # stacked
# df2.plot.bar(stacked=True)

# # horizontal
# df2.plot.barh(stacked=True)

##############################################################################################################
# box plot

# df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
# color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
# df.plot.box(color=color, vert=False)

##############################################################################################################
# area plot

# df = pd.DataFrame(np.random.randint(5, size=(20,4)).astype('uint8'), columns=['a', 'b', 'c', 'd'])
# df.plot.area()

##############################################################################################################
# scatter plot

df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
# df.plot.scatter(x='a', y='b');

# ax = df.plot.scatter(x='a', y='b', color='DarkBlue', label='Group 1');
# df.plot.scatter(x='c', y='d', color='DarkGreen', label='Group 2', ax=ax);

# df.plot.scatter(x='a', y='b', c='c', s=50)
# df.plot.scatter(x='a', y='b', s=df['c'] * 200);

##############################################################################################################
# hexagonal bin plot

# df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
# df['b'] = df['b'] + np.arange(1000)
# # df['z'] = np.random.uniform(0, 3, 1000)
# df.plot.hexbin(x='a', y='b', gridsize=25)

##############################################################################################################
# pie plot

# series = pd.Series(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], name='')
# series.plot.pie(labels=['AA', 'BB', 'CC', 'DD'],
#                 autopct='%.2f', fontsize=18, figsize=(10, 10))

##############################################################################################################
# scatter matrix plot

# from pandas.tools.plotting import scatter_matrix
# df = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])
# scatter_matrix(df, alpha=0.7, figsize=(8, 8), diagonal='kde', grid=False)


##############################################################################################################
# RadViz

# from pandas.tools.plotting import radviz
# df['e'] = np.random.randint(2, size=50)
# radviz(df, 'e')


plt.show()