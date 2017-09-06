"""
Task5, RF prediction
"""

import json
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(76543)

def save_answers5(accuracy):
     with open("cooking_LDA_pa_task5.txt", "w") as fout:
        fout.write(str(accuracy))

# load data and models 

with open("recipes.json") as f:
    recipes = json.load(f)

print("load data")

texts = [recipe["ingredients"] for recipe in recipes]
dictionary2 = corpora.Dictionary(texts) 
freq_tokens = []
for k, v in dictionary2.dfs.items():
    if v>4000: freq_tokens.append(k)
dictionary2.filter_tokens(bad_ids=freq_tokens)
corpus2 = [dictionary2.doc2bow(text) for text in texts]

model2 = LdaModel.load("model2")

# prepare X, y

print("prepare X,y") 

y = []
for recipe in recipes:
    y.append(recipe['cuisine'])

theta = model2.get_document_topics(bow=corpus2)
X = np.zeros((model2.num_updates, model2.num_topics))
for i, document in enumerate(theta):
    for theme in document:
        X[i, theme[0]-1] = theme[1]

# learn RF
print("learn RF")
rf = RandomForestClassifier(n_estimators=100)
score = cross_val_score(cv=3, estimator=rf, scoring='accuracy', X=X, y=y)

save_answers5(score.mean())
