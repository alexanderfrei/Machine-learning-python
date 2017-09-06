"""
Task3, coherence 
"""

import json
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import copy

np.random.seed(76543)

def save_answers3(coherence, coherence2):
    with open("cooking_LDA_pa_task3.txt", "w") as fout:
        fout.write(" ".join(["%3f"%el for el in [coherence, coherence2]]))

with open("recipes.json") as f:
    recipes = json.load(f)

# load data and models 

model = LdaModel.load('model1')
texts = [recipe["ingredients"] for recipe in recipes]
dictionary = corpora.Dictionary(texts) 
corpus = [dictionary.doc2bow(text) for text in texts] 

# create new dictionary 

print("create new dictionary")
dictionary2 = copy.deepcopy(dictionary)
freq_tokens = []
for k, v in dictionary2.dfs.items():
    if v>4000: freq_tokens.append(k)
dictionary2.filter_tokens(bad_ids=freq_tokens)
corpus2 = [dictionary2.doc2bow(text) for text in texts]

# learn model2 

print("learn model 2")
model2 = LdaModel(corpus=corpus2, id2word=dictionary2, num_topics=40, passes=5)
model2.save('model2')

# compute coherence 

print("compute coherence")

top1 = model.top_topics(corpus=corpus)
top2 = model2.top_topics(corpus=corpus2)

arr1 = np.array(top1)
arr2 = np.array(top2)

save_answers3(arr1[:,1].mean(), arr2[:,1].mean())
