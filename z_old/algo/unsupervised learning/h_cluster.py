from math import sqrt
import numpy as np
import pandas as pd


def pearson(v1, v2):
    sum1 = sum(v1)
    sum2 = sum(v2)
    sum1_sq = sum([v ** 2 for v in v1])
    sum2_sq = sum([v ** 2 for v in v2])
    pr_sum = sum([v1[i] * v2[i] for i in range(len(v1))])

    num = pr_sum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1_sq - (sum1 ** 2) / len(v1)) * (sum2_sq - (sum2 ** 2) / len(v1)))

    if den == 0: return 0
    return 1 - num / den


class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance


def hcluster(rows, distance=pearson):
    distances = {}
    current_clust_id = -1
    clust = [bicluster(data.iloc[i, :], id=i) for i in range(data.shape[0])]

    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # check all pairs and search pair with minimal distance
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # save calculated distances in cache
                if (clust[i].id, clust[j].id) not in distances:
                    distances[clust[i].id, clust[j].id] = distance(clust[i].vec, clust[j].vec)
                d = distances[clust[i].id, clust[j].id]
                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        # get average for lowest clusters
        merge_vec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0 \
                     for i in range(len(clust[0].vec))]

        # new cluster
        new_cluster = bicluster(merge_vec,
                                left=clust[lowestpair[0]],
                                right=clust[lowestpair[1]],
                                distance=closest,
                                id=current_clust_id)

        current_clust_id -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(new_cluster)

    return clust[0]


def dendro(clust, labels=None, n=0):
    for i in range(n):
        print(' ', end='')
    if clust.id < 0:
        print('-')
    else:
        if labels is None: print(clust.id)
        else: print(labels[clust.id])
    if clust.left is not None: dendro(clust.left, labels=labels, n=n + 1)
    if clust.right is not None: dendro(clust.right, labels=labels, n=n + 1)



if __name__ == "__main__":

    data = pd.read_csv('rss_word_count.csv', header=0, delimiter=',')
    rownames = list(data.loc[:, 'blog'])
    colnames = data.iloc[0, 1:]
    data = data.iloc[:, 1:]

    hclust = hcluster(data)
    dendro(hclust, labels=rownames)

    X = np.array(data)
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    Z = linkage(X, method='average', metric='correlation')
    dendrogram(Z, leaf_rotation=90, labels=rownames)
    plt.show()