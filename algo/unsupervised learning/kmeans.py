import random
from h_cluster import pearson as pearson
import pandas as pd

def kmeans_clust(data, distance=pearson, k=4):

    # find min and max points
    ranges = [(data.iloc[:,j].min(), data.iloc[:,j].max()) for j in range(data.shape[1])]
    # random centroids
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(data.shape[1])] for j in range(k)]
    # start iterating
    lastmatches = None
    for it in range(100):
        print('Iteration {}..'.format(it + 1))
        bestmatches = [[] for i in range(k)]

        # find nearest centroids for each line
        for j in range(data.shape[0]):
            row = data.iloc[j, :]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row): bestmatch = i
            bestmatches[bestmatch].append(j)

        # success if no changes
        print(bestmatches)
        print(lastmatches)
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # compute new cluster centers
        for i in range(k):
            avgs = [0.0] * data.shape[1]
            if len(bestmatches[i]) > 0:
                for obj_id in bestmatches[i]:
                    for ft in range(data.shape[1]):
                        avgs[ft] += data.iloc[obj_id, ft]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs

    return bestmatches

data = pd.read_csv('rss_word_count.csv', header=0, delimiter=',')
rownames = list(data.loc[:, 'blog'])
# colnames = data.iloc[0, 1:]
data = data.iloc[:, 1:]

clusters = kmeans_clust(data)
for c in clusters:
    print('')
    for feed in c:
        print(rownames[feed],end=' ')

