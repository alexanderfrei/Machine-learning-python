# example of Boruta feature selection

from ml_tools.tools import *
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

X = pickle_load('./input/X').values
y = pickle_load('./input/y').values

rfc = RandomForestClassifier(n_jobs=2, class_weight='balanced', max_depth=6)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, max_iter=150)
boruta_selector.fit(X, y)
pickle_dump(boruta_selector, './input/boruta_selector')
