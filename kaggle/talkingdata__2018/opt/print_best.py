# coding: utf-8
from hyperopt import space_eval
import pickle
import sys 

try:
    path = sys.argv[1]+'/'
except IndexError:
    path = ''

with open(path+'best.pkl', 'rb') as f: best = pickle.load(f)
with open(path+'space.pkl', 'rb') as f: space = pickle.load(f)
with open(path+'max_score.pkl', 'rb') as f: max_score = pickle.load(f)

best_params = space_eval(space, best)
print(best_params)

[print('{%.4f}' % score) for score in max_score]