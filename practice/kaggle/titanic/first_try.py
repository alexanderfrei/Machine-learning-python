import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# os.getcwd()

gender_sub = pd.read_csv("data/gender_submission.csv", header=0)
train = pd.read_csv("data/train.csv", header=0)
test = pd.read_csv("data/test.csv", header=0)

# print(pd.crosstab(gender_sub.Survived,gender_sub.Survived))
# df = pd.DataFrame({'col1':np.random.randn(100),'col2':np.random.randn(100)})
# df.hist(layout=(1,2))
#

# df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
#                     'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
train.diff().hist(alpha=0.5, color='k')
plt.show()
