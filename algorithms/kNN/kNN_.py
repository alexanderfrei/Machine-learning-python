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


def gaussian(dist, sigma=10.0):
    return math.e ** (-dist ** 2 / (2 * sigma ** 2))








#
# print(wine_price(95, 3))
data = wine_set1()
# print(get_distances(data))
# print(get_distances(data, (0, 300)))
print(knn_estimate(data, (60, 3)))
