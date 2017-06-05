# TODO Добавить BFS + DFS
# TODO переписать в функцию

import pandas as pd
import math

# url reference matrix
urls = {
    "a": [0,1,1,1],
    "b": [1,0,0,1],
    "c": [0,0,0,1],
    "d": [1,0,0,0],
}

urls = {
    "a": [0,1,1,1],
    "b": [1,0,0,1],
    "c": [0,0,0,1],
    "d": [0,0,0,0],
}

# urls = {
#     "a": [0,0,0,1],
#     "b": [1,0,0,0],
#     "c": [0,1,0,0],
#     "d": [0,0,1,0],
# }

urls = pd.DataFrame.from_dict(urls, orient="index")
urls = urls.sort_index()
urls_list = urls.index.values

c_dict = {}
for index, name in zip(urls.axes[1], urls_list):
    c_dict[index] = name
urls = urls.rename(columns=c_dict)

reference_num = []
page_rank = []

for i, url in enumerate(urls_list):
    # check sink
    if urls.loc[url, :].sum() == 0:
        urls.loc[url, :] = 1
    urls.loc[url, url] = 0
    reference_num.append(urls.loc[url, :].sum())
    page_rank.append(1)

n_iter = 1
while n_iter < 20:
    pr = []
    for i, url in enumerate(urls_list):
        pr.append(0.15 + 0.85 * sum(urls.loc[url, :] * page_rank / reference_num))

    ch = math.fabs(sum([page_rank[j] - pr[j] for j in range(len(urls_list))]))
    print(n_iter, pr, ch)
    if ch < 0.01:
        page_rank = pr
        break

    page_rank = pr
    n_iter += 1

print(page_rank)

