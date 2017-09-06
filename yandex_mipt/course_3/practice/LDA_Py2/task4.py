"""
Task4, alpha=1
"""

import json
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel

np.random.seed(76543)

def save_answers4(count_model2, count_model3):
    with open("cooking_LDA_pa_task4.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [count_model2, count_model3]]))
        
with open("recipes.json") as f:
    recipes = json.load(f)

# load data and models 

print("load data")

texts = [recipe["ingredients"] for recipe in recipes]
dictionary2 = corpora.Dictionary(texts) 
freq_tokens = []
for k, v in dictionary2.dfs.items():
    if v>4000: freq_tokens.append(k)
dictionary2.filter_tokens(bad_ids=freq_tokens)
corpus2 = [dictionary2.doc2bow(text) for text in texts]

model2 = LdaModel.load("model2")

# learn model 3 
print("learn model 3")

model3 = LdaModel(corpus=corpus2, id2word=dictionary2, passes=5, alpha=1)
model3.save('model3')

# compute answers
print("count len documents from T-D matrix")

theta = model2.get_document_topics(bow=corpus2, minimum_probability=0.01)
counter_model2 = 0
for document in theta:
    counter_model2 += len(document)

theta = model3.get_document_topics(bow=corpus2, minimum_probability=0.01)
counter_model3 = 0
for document in theta:
    counter_model3 += len(document)

save_answers4(counter_model2, counter_model3)

