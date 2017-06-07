import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(url, names=names)
X, Y = df.iloc[:, 0:8], df.iloc[:, 8]


# SelectKBest
from sklearn.feature_selection import SelectKBest, chi2
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

for score, col in zip(fit.scores_, names):
    print(score, col)

features = fit.transform(X)
print(features[0:5,:])


# RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)
print(fit.n_features_, fit.support_, fit.ranking_)
print(df.loc[0:5,fit.support_])


# Feature Extraction with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
fit = pca.fit(X)
features = fit.transform(X)
print("Explained Variance: {}".format(fit.explained_variance_ratio_))
print(features[0:5,:])


# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

fi = []
for i,n in zip(model.feature_importances_, names):
    fi.append((i, n))
sorted(fi, key=lambda x: x[0], reverse=True)

