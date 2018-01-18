from ml_tools.tools import *
from ml_tools.preprocessing import cut
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('input/train.csv') # .iloc[0:20000,:]
test_df = pd.read_csv('input/test.csv') # .iloc[0:20000,:]

train_features = [
	"ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
    "ps_car_11_cat",
]

# count missings
train_df['missings'] = np.sum(train_df == -1, axis=1)
test_df['missings'] = np.sum(test_df == -1, axis=1)
train_features = train_features + ['missings']

# bin difference
bin_ = [c for c in train_features if 'bin' in c]

frow = np.array(train_df.loc[0, bin_])
arr = np.array(train_df[bin_])
train_df['bin_diff'] = (arr != frow).sum(axis=1)

frow = np.array(test_df.loc[0, bin_])
arr = np.array(test_df[bin_])
test_df['bin_diff'] = (arr != frow).sum(axis=1)

train_features += ["bin_diff"]

cat = [c for c in train_features if '_cat' in c]
bin = [c for c in train_features if '_bin' in c and not c in cat]
ind = [c for c in train_features if '_ind' in c and not c in cat and not c in bin]
ind = list(set(train_features).difference(ind+cat+bin)) + ind

# fill missing
cats = [c for c in train_features if '_cat' in c]
vals = list(set(train_features).difference(bin_+cats+['bin_diff']))
vals_miss = ['ps_car_11', 'ps_car_12']

train_df.loc[:, vals_miss] = train_df.loc[:, vals_miss].replace(-1, np.nan)
test_df.loc[:, vals_miss] = test_df.loc[:, vals_miss].replace(-1, np.nan)

vals_mean = train_df[vals_miss].median()
train_df.loc[:, vals_miss] = train_df.loc[:, vals_miss].fillna(vals_mean)
test_df.loc[:, vals_miss] = test_df.loc[:, vals_miss].fillna(vals_mean)

# cut

to_cut = ['ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_reg_03']
for var in to_cut:
    print(var)
    train_df, test_df = cut(train_df, test_df, var)

# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
    ('ps_ind_14', 'ps_car_15'),
    ('ps_reg_02', 'ps_car_15'),
    ('ps_car_15', 'ps_ind_05_cat'),
    ('ps_ind_14', 'ps_ind_05_cat'),
    ('ps_ind_01', 'ps_ind_14'),
    ('ps_ind_15', 'ps_ind_14'),
    ('ps_car_15', 'ps_ind_14'),
    ('ps_reg_01', 'ps_ind_14'),
    ('ps_ind_14', 'ps_car_12qcut'),
    ('ps_ind_14', 'ps_car_13qcut'),
    ('ps_ind_15', 'ps_car_12qcut'),
    ('ps_car_15', 'ps_car_14qcut'),
    ('ps_ind_14', 'ps_car_14qcut'),
    ('ps_ind_15', 'ps_car_14qcut'),
    ('ps_car_13qcut', 'ps_ind_05_cat'),
    ('ps_car_13qcut', 'ps_ind_06_bin'),
    ('ps_car_14qcut', 'ps_ind_14'),
    ('ps_reg_03qcut', 'ps_ind_14'),
    ('ps_reg_02', 'ps_car_13qcut'),
    ('ps_reg_03qcut', 'ps_car_15'),
    ('ps_reg_02', 'ps_reg_03qcut')
]

id_test = test_df['id'].values
id_train = train_df['id'].values
y = train_df['target']

for n_c, (f1, f2) in enumerate(combs):
    print(f1 + ' ' + f2)

    name1 = f1 + "_plus_" + f2 + "_cat"
    train_df[name1] = train_df[f1].apply(lambda x: str(x)) + "_" + train_df[f2].apply(lambda x: str(x))
    test_df[name1] = test_df[f1].apply(lambda x: str(x)) + "_" + test_df[f2].apply(lambda x: str(x))

    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
    train_df[name1] = lbl.transform(list(train_df[name1].values))
    test_df[name1] = lbl.transform(list(test_df[name1].values))
    train_features.append(name1)

X = train_df[train_features]
X_test = test_df[train_features]

# Feature engineering
X.loc[:, 'ps_car_11x14'] = X.ps_car_11 * X.ps_car_14
X.loc[:, 'ps_ind_01x_car_14'] = X.ps_ind_01 * X.ps_car_14

X_test.loc[:, 'ps_car_11x14'] = X_test.ps_car_11 * X_test.ps_car_14
X_test.loc[:, 'ps_ind_01x_car_14'] = X_test.ps_ind_01 * X_test.ps_car_14



# dump result
path = "../../input/"
pickle_dump(X, path+'X')
pickle_dump(y, path+'y')
pickle_dump(X_test, path+'X_test')
pickle_dump(id_train, path+'id_train')
pickle_dump(id_test, path+'id_test')



### segmentation (optionally)

# onehotencoding
X_scl = X.copy()
X_test_scl = X_test.copy()
f_cats = [c for c in X.columns if '_cat' in c]

for column in f_cats:
    temp = pd.get_dummies(pd.Series(X_scl[column]), prefix=column)
    X_scl = pd.concat([X_scl, temp], axis=1)
    X_scl = X_scl.drop([column], axis=1)

for column in f_cats:
    temp = pd.get_dummies(pd.Series(X_test_scl[column]), prefix=column)
    X_test_scl = pd.concat([X_test_scl, temp], axis=1)
    X_test_scl = X_test_scl.drop([column], axis=1)

f_cats = [c for c in X_scl.columns if '_cat' in c]
f_oth = [c for c in X.columns if not '_cat' in c]
X_cat = X_scl[f_cats]

chi2_pval = chi2(X_cat, y)[1]
X_scl = pd.concat([X_scl[f_oth], X_cat.loc[:, chi2_pval < 0.05]], axis=1)

joint_cols = list(set(X_test_scl.columns).intersection(X_scl.columns))
X_scl = X_scl[joint_cols]
X_test_scl = X_test_scl[joint_cols]

f_oth = [c for c in X.columns if not '_cat' in c]

imputer = Imputer(strategy='median')

X_scl[f_oth] = X_scl[f_oth].replace(-1, np.nan)
X_test_scl[f_oth] = X_test_scl[f_oth].replace(-1, np.nan)
X_scl[f_oth] = imputer.fit_transform(X_scl[f_oth])
X_test_scl[f_oth] = imputer.transform(X_test_scl[f_oth])

# scaling

f_oth = [c for c in X_scl.columns if not '_cat' in c]
scaler = StandardScaler()
X_scl[f_oth] = scaler.fit_transform(X_scl[f_oth])
X_test_scl[f_oth] = scaler.transform(X_test_scl[f_oth])

# simple segmentation with kmeans

def kmeans_(train, test, n, n_jobs=3):
    """ kmeans 
    train, test: DataFrames, scaled by features 
    """
    clu = KMeans(n_clusters=n, n_jobs=n_jobs).fit(train)
    train['clu'+n] = clu.labels_
    test['clu'+n] = clu.predict(test)
    return train, test

X_scl, X_test_scl = kmeans_(X_scl, X_test_scl, 5)
X_scl, X_test_scl = kmeans_(X_scl, X_test_scl, 10)
X_scl, X_test_scl = kmeans_(X_scl, X_test_scl, 25)