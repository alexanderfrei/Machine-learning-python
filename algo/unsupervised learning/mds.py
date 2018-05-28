
from math import sqrt
import random
from h_cluster import pearson as pearson
import pandas as pd

data = pd.read_csv('rss_word_count.csv', header=0, delimiter=',')
rownames = list(data.loc[:, 'blog'])
# colnames = data.iloc[0, 1:]
data = data.iloc[:, 1:]

# distance matrix
rate = 0.01
distance = pearson
n = data.shape[0]
dist = [[distance(data.iloc[i,:], data.iloc[j,:]) for j in range(n)]
            for i in range(n)]

# place object in random location
loc = [[random.random(), random.random()] for i in range(n)]
fake_dist = [[0.0 for j in range(n)] for i in range(n)]
last_error = None

for m in range(1000):
    for i in range(n):
        for j in range(n):
            fake_dist[i][j] = sqrt(sum([pow(loc[i][x] - loc[j][x],2) for x in range(2)]))
    # move points
    grad = [[0.0,0.0] for i in range(n)]
    total_error = 0.0
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            if dist[j][k]:
                error_term = (fake_dist[j][k] - dist[j][k]) / dist[j][k]
            if dist[j][k] == 0:
                error_term = fake_dist[j][k]
            print(m, k, j, dist[j][k])
            grad[k][0] += ((loc[k][0] - loc[j][0]) / fake_dist[j][k]) * error_term
            grad[k][1] += ((loc[k][1] - loc[j][1]) / fake_dist[j][k]) * error_term
            total_error += abs(error_term)
    print(total_error)

    if last_error and last_error < total_error:
        break
    last_error = total_error

    for k in range(n):
        loc[k][0] += rate * grad[k][0]
        loc[k][1] += rate * grad[k][1]

print(loc)

from PIL import Image
from PIL import ImageDraw
img = Image.new('RGB', (1000,1000),(255,255,255))
draw = ImageDraw.Draw(img)
for i in range(len(loc)):
    x = (loc[i][0] + 0.5) * 1000
    y = (loc[i][1] + 0.5) * 1000
    draw.text((x,y), rownames[i],(0,0,0))
img.save('mds.jpg','JPEG')

