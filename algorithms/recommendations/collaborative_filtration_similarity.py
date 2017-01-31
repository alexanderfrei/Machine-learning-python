from operator import itemgetter

critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                         'The Night Listener': 3.0},
           'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 3.5},
           'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},
           'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},
           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 2.0},
           'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
           'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}

users = {'user1': ['Google','Yandex','Mail.Ru'],
         'user2': ['Bing','Yahoo','Google'],
         'user3': ['Naver','Baidu','Google'],
         'user4': ['DuckDuckGo'],
         'user5': ['Naver','Google','DuckDuckGo','Baidu'],
         'user6': ['Bing','Yahoo','Google','Aol']
        }

def sim_pearson(prefs, p1, p2):
    from math import sqrt
    # collaborative items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1
    # If no collaborative items return 0
    if len(si) == 0:
        return 0
    n = len(si)
    # sum of preference
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    # sum of product
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
    # sum of squares
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
    # calculate Pearson coefficient
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0: return 0
    r = num / den
    return r


def top_matches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


def get_recommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs:
        # not me
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # if positive similarity
        if sim <= 0:
            continue
        for item in prefs[other]:
            # evaluate movie I didn't see
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # sum of similarity weights
                simSums.setdefault(item, 0)
                simSums[item] += sim

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort(reverse=True)
    return rankings


def transform_prefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result


def calculate_similar_items(prefs, n=10):
    # create dict of top top-n similar items
    result = {}
    # items preference matrix
    item_prefs = transform_prefs(prefs)
    c = 0
    for item in item_prefs:
        # logging
        c += 1
        if c % 100 == 0:
            print("%d / %d".format(c, len(item_prefs)))
        # list of scores (tuples)
        scores = top_matches(item_prefs, item, n=n, similarity=sim_pearson)
        result[item] = scores
    return result


def get_recommended_items(prefs, item_match, user):
    user_ratings = prefs[user]
    scores = {}
    total_sim = {}
    # iterating by evaluated movies
    for (item, rating) in user_ratings.items():
        # iterating by movies, similar to current movie
        for (similarity, item2) in item_match[item]:
            # ignore evaluated movie
            if item2 in user_ratings:
                continue
            # weight(sim) * rating
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # sum of weights(sim)
            total_sim.setdefault(item2, 0)
            total_sim[item2] += similarity
            rankings = [(score / total_sim[item], item) for item, score in scores.items()]
    rankings.sort(reverse=True)
    return rankings


def transform_dlist(d_lst):
    items_dict = {}
    for d_key in d_lst:
        for item in d_lst[d_key]:
            items_dict.setdefault(item, [])
            items_dict[item].append(d_key)
    return items_dict


def sim_tanimoto(d_list, sim_key):
    similarity = {}
    a = len(d_list[sim_key])
    for d_key in d_list:
        if d_key == sim_key:
            continue
        b = len(d_list[d_key])
        c = 0
        for item in d_list[d_key]:
            if item in d_list[sim_key]:
                c += 1
        similarity[d_key] = c / (a + b - c)
    return similarity

print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))
print(top_matches(critics, 'Toby'))
print(get_recommendations(critics, 'Toby'))

movies = transform_prefs(critics)
print(top_matches(movies, 'Superman Returns'))

sim_movies_base = calculate_similar_items(critics, 5)
print(get_recommended_items(critics, sim_movies_base, 'Toby'))

engines = transform_dlist(users)
sim_engines = {}
for engine in engines:
    sim_engine = sorted(sim_tanimoto(engines, engine).items(), key=itemgetter(1), reverse=True)
    sim_engines[engine] = sim_engine
print(sim_engines)