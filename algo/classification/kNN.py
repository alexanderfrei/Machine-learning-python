from random import random, randint
import math


def wine_price(rating, age):
    peak_age = rating - 50
    # Вычислить цену в зависимости от рейтинга
    price = rating / 2
    if age > peak_age:
        # Оптимальный возраст пройден, через 5 лет испортится
        price *= 5 - (age - peak_age)
    else:
        # Увеличивать до пятикратной начальной цены по мере
        # приближения к оптимальному возрасту
        price *= 5 * ((age + 1) / peak_age)
    if price < 0:
        price = 0
    return price


def wine_set1():
    rows = []
    for i in range(300):
        rating = random() * 50 + 50
        age = random() * 50
        price = wine_price(rating, age)
        price *= (random() * 0.4 + 0.8)
        rows.append({'input': (rating, age), 'result': price})
    return rows


def euclidean(v1, v2):
    d = 0.0
    for i in range(len(v1)):
        d += (v1[i] - v2[i]) ** 2
    return math.sqrt(d)


def get_distances(data,vec1):
    distance_list = []
    for i in range(len(data)):
        vec2 = data[i]['input']
        distance_list.append((euclidean(vec1, vec2), i))
    distance_list.sort()
    return distance_list


def knn_estimate(data,vec1,k=3):
    # Получить отсортированный список расстояний
    dlist = get_distances(data, vec1)
    avg = 0.0
    # Усреднить по первым k образцам
    for i in range(k):
        idx = dlist[i][1]
    avg += data[idx]['result']
    avg /= k
    return avg


def inverse_weight(dist, num=1.0, const=0.1):
    return num / (dist + const)


def subtract_weight(dist, const=1.0):
    if dist > const:
        return 0
    else:
        return const - dist


def gaussian(dist, sigma=1.0):
    return math.e ** (-dist ** 2 / (2 * sigma ** 2))
# / math.sqrt(2 * math.pi * sigma ** 2)


def weighted_knn(data,vec1,k=5,weight_f=gaussian):

    dlist = get_distances(data, vec1)
    avg = 0.0
    total_weight = 0

    for i in range(k):
        dist = dlist[i][0]
        idx = dlist[i][1]
        weight = weight_f(dist)
        avg += weight * data[idx]['result']
        total_weight += weight
    avg /= total_weight
    return avg

def wine_set3( ):
    rows = wine_set1()
    for row in rows:
        if random() < 0.5:
            row['result'] *= 0.6
    return rows

def probguess(data,vec1,low,high,k=10,weightf=gaussian):
    dlist = get_distances(data, vec1)
    nweight = 0.0
    tweight = 0.0
    for i in range(k):
        dist = dlist[i][0]
        idx = dlist[i][1]
        weight = weightf(dist)
        v = data[idx]['result']
        print(v,low,high)
        if v >= low and v <= high:
            nweight += weight
        tweight += weight
    if tweight == 0: return 0
    return nweight / tweight

data = wine_set3()
print(probguess(data, (80,10), 40, 80))
# print(data)
# print(get_distances(data, (60, 10)))
# print(knn_estimate(data, (99, 5)))
# print(weighted_knn(data, (99, 5)))
