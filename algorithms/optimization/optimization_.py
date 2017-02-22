# 3. Optimization
import math
import random
import pandas as pd
import numpy as np

def cost(dist, start, route):
    route_len = int(dist.loc[start,route[0]])
    route_len += int(dist.loc[start, route[-1]])
    for i in range(1,len(route)):
        route_len += int(dist.loc[route[i],route[i-1]])
    return route_len

def random_opt(dist, start, goal, costf):
    best = 1000000
    best_decision = None
    for i in range(10000):
        decision = np.random.permutation(goal)
        cost = costf(dist, start, decision)
        if cost < best:
            best = cost
            best_decision = decision
    return best_decision

def swap(ls, c1, c2):
    tls = ls.copy()
    tmp = tls[c1]
    tls[c1] = tls[c2]
    tls[c2] = tmp
    return tls

############################################################

# dist = pd.read_csv("table.russia.csv")
# cities = dist.iloc[0, 1:]
# idict = {}
# for index, name in zip(dist.axes[0], dist.iloc[:, 0]):
#     idict[index] = name
# dist = dist.rename(idict)
#
# route = ["Зеленоград", "Иваново", "Ижевск", "Уссурийск", "Челябинск", "Шахты", "Сургут", "Череповец", "Сочи"]
# start = "Москва"
#
# result_rnd = random_opt(dist, start, route, cost)
# print(cost(dist, start, result_rnd), result_rnd, sep='\n')

############################################################

def random_wine_opt(wine, costf):
    best = 10000000
    best_weights = None
    for i in range(10000):
        weights = np.random.choice(range(-10,10), wine.shape[1] - 1)
        cost = costf(wine, weights)
        if cost < best:
            best = cost
            best_weights = weights
    return best_weights

def hill_climb_opt(data, costf, rate=0.05):
    weights = list(np.random.choice(range(-10,10), data.shape[1] - 1))
    while 1:
        neighbors = []
        # neighbors
        for i in range(data.shape[1] - 1):
            n_weights = weights.copy()
            n_weights[i] -= rate
            neighbors.append(n_weights.copy())
            n_weights[i] += rate * 2
            neighbors.append(n_weights.copy())
        current = costf(data, weights)
        best = current
        for n in range(len(neighbors)):
            cost = costf(data, neighbors[n])
            if cost < best:
                best = cost
                weights = neighbors[n]
        print(best)
        # stop if no improvement
        if best == current:
            break
    return weights

def annealing_opt(data, costf, T=10 ** 12, cool=0.995, step=0.05):
    weights = np.random.choice(range(-10, 10), wine.shape[1] - 1)
    weights = weights.astype(float)
    while T > 0.1:
        # random index
        i = random.randint(0, weights.shape[0] - 1)
        # random direction
        dir = random.choice([-step,step])

        n_weights = weights.copy()
        n_weights[i] += dir

        last = costf(data, weights)
        new = costf(data, n_weights)
        # choice probability
        p = math.e ** ((-new-last)/T)
        if (new < last or random.random() < p):
            weights = n_weights
        print(last, new, p, T)
        # cool
        T = T * cool

    return weights


def cost_wine(wine, weights):
    # assume last column as prediction
    y_est = wine.iloc[:, :-1] * weights
    return sum((wine.iloc[:, wine.shape[1] - 1] - y_est.sum(axis=1)) ** 2)

wine = pd.read_csv("winequality-red.csv", sep = ';')

# result_random = random_wine_opt(wine, cost_wine)
# print(cost_wine(wine, result_random), result_random, sep='\n')

# result_hill = hill_climb_opt(wine, cost_wine)
# print(cost_wine(wine, result_hill), result_hill, sep='\n')

result_ann = annealing_opt(wine, cost_wine)
print(cost_wine(wine, result_ann), result_ann, sep='\n')

