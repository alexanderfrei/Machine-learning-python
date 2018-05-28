
# Optimization
import math
import random
import pandas as pd
import numpy as np

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


def genetic_wine_opt(wine, costf, popsize=100, rate=0.05, mutprob=0.5, elite=0.25, maxiter=100):

    def mutate(weights):
        i = random.randint(0, wine.shape[1] - 2)
        if random.random() >= 0.5:
            weights[i] += rate
        else:
            weights[i] -= rate
        return weights

    def crossover(w1, w2):
        i = random.randint(1, wine.shape[1] - 3)
        return w1[0:i] + w2[i:]

    pop = []
    for i in range(popsize*5):
        weights = [random.uniform(-5,5) for i in range(wine.shape[1]-1)]
        pop.append(weights)
    top_elite = int(elite*popsize)
    for i in range(maxiter):
        scores = [(costf(wine, v), v) for v in pop]
        scores.sort()
        ranked = [v for (s,v) in scores]
        # best weights
        pop = ranked[0:top_elite]
        # mutate + crossover from best weights
        while len(pop)<popsize:
            if random.random() < mutprob:
                c = random.randint(0, top_elite)
                pop.append(mutate(ranked[c]))
            else:
                c1 = random.randint(0, top_elite)
                c2 = random.randint(0, top_elite)
                pop.append(crossover(ranked[c1],ranked[c2]))
        print(scores[0][0], scores[0][1])

    return scores[0][1]

def cost_wine(wine, weights):
    # assume last column as prediction
    y_est = wine.iloc[:, :-1] * weights
    return sum((wine.iloc[:, wine.shape[1] - 1] - y_est.sum(axis=1)) ** 2)

wine = pd.read_csv("winequality-red.csv", sep = ';')

# result_random = random_wine_opt(wine, cost_wine)
# print(cost_wine(wine, result_random), result_random, sep='\n')

# result_hill = hill_climb_opt(wine, cost_wine)
# print(cost_wine(wine, result_hill), result_hill, sep='\n')

# result_ann = annealing_opt(wine, cost_wine)
# print(cost_wine(wine, result_ann), result_ann, sep='\n')

# result = genetic_wine_opt(wine, cost_wine)
# print(cost_wine(wine, result), result, sep='\n')



############################################################

def cost(dist, start, route):
    route_len = int(dist.loc[start, route[0]])
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

def genetic_dist_opt(dist, start, route, costf, popsize=150, mutprob=1, elite=0.5, maxiter=50):

    def mutate(decision):
        i1 = random.randint(0, len(decision)-1)
        i2 = i1
        while i2 == i1:
            i2 = random.randint(0, len(decision)-1)
        decision = swap(decision, i1, i2)
        return decision

    def crossover(d1, d2):
        unique = []
        decision = [0] * (len(d1))
        for i in range(len(d1)):
            # fixing common values
            if d1[i] == d2[i]:
                decision[i] = d1[i]
            else:
                if d1[i] not in unique:
                    unique.append(d1[i])
                if d2[i] not in unique:
                    unique.append(d2[i])
        random.shuffle(unique)
        for i in range(len(decision)):
            if not decision[i]:
                decision[i] = unique.pop()
        return decision

    pop = []
    for i in range(popsize*5):
        decision = random.sample(route, len(route))
        pop.append(decision)
    top_elite = int(elite*popsize)
    for i in range(maxiter):
        scores = [(costf(dist, start, v), v) for v in pop]
        scores.sort()
        ranked = [v for (s,v) in scores]
        # best decisions
        pop = ranked[0:top_elite]
        # mutate + crossover from best decisions
        while len(pop) < popsize:
            if random.random() < mutprob:
                c = random.randint(0, top_elite)
                pop.append(mutate(ranked[c]))
            else:
                c1 = random.randint(0, top_elite)
                c2 = random.randint(0, top_elite)
                pop.append(crossover(ranked[c1],ranked[c2]))
        print(scores[0][0], scores[0][1])

    return scores[0][1]


dist = pd.read_csv("table.russia.csv")
cities = dist.iloc[0, 1:]
idict = {}
for index, name in zip(dist.axes[0], dist.iloc[:, 0]):
    idict[index] = name
dist = dist.rename(idict)

route = ["Зеленоград", "Иваново", "Ижевск", "Уссурийск", "Челябинск", "Шахты", "Сургут", "Череповец", "Сочи"]
start = "Москва"

# result_rnd = random_opt(dist, start, route, cost)
# print(cost(dist, start, result_rnd), result_rnd, sep='\n')

# result = genetic_dist_opt(dist, start, route, cost)
# print(cost(dist, start, result), result, sep='\n')