import seaborn as sns
import pandas as pd
import numpy as np

# create freq matrix
def cross_freq(arr):
    shape = arr.shape[1]
    f = np.zeros((shape, shape))
    for j in range(0, shape):
        for i in range(j, shape):
            ind = np.where(np.logical_and(~np.isnan(arr[..., j]), ~np.isnan(arr[..., i])))
            f[j, i], f[i, j] = ind[0].shape[0], ind[0].shape[0]
    return f

engines = pd.read_csv('engines.csv', header=0, delimiter=',')
f = np.array(engines)
f = cross_freq(f)
e = pd.DataFrame(f)

names = {}
for i, col in enumerate(engines.columns):
    names[i] = col

e = e.rename(names)
e = e.rename(columns=names)
print(e)

mask = np.zeros_like(e)
mask[np.triu_indices_from(mask)] = True

sns.set_style("white")
hmap = sns.heatmap(e, cmap=sns.cubehelix_palette(8, start=1, dark=0.1, light=.85, as_cmap=True),
                 linewidths=.3,
                 mask = mask,
                 vmax = 100,
                 vmin = 1)
sns.plt.yticks(rotation=0)
sns.plt.xticks(rotation=90)
sns.plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)
# sns.plt.savefig('heatmap.pdf')
sns.plt.show()
