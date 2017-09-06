import json
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel

np.random.seed(76543)

def save_answers1(ls):
    with open("cooking_LDA_pa_task1.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in ls]))

with open("recipes.json") as f:
    recipes = json.load(f)

texts = [recipe["ingredients"] for recipe in recipes]
dictionary = corpora.Dictionary(texts) 
corpus = [dictionary.doc2bow(text) for text in texts] 

model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=40, passes=5)
top10 = model.top_topics(corpus=corpus, num_words=10)

counter = [0] * 6
token_list = ["salt", "sugar", "water", "mushrooms", "chicken", "eggs"]
for topic in top10:
    for token in topic[0]:
        for i in range(len(token_list)):
            if token[1] == token_list[i]: counter[i] += 1

save_answers1(counter)
model.save('model1')
