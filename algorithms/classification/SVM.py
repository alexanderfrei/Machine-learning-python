# TODO check + descriptions

import math
class match_row:
    def __init__(self, row, allnum=False):
        if allnum:
            self.data = [float(row[i]) for i in range(len(row)-1)]
        else:
            self.data = row[0:len(row)-1]
        self.match = int(row[len(row)-1])

def load_match(f, allnum=False):
    rows = []
    with open(f) as file:
        for line in file:
            rows.append(match_row(line.split(','), allnum))
    return rows

ages_only = load_match('agesonly.csv', allnum=True)
match_maker = load_match('matchmaker.csv')

# print(ages_only[0].match)
# print(match_maker)

from pylab import *
def plot_age_matches(rows):
    xdm, ydm = [r.data[0] for r in rows if r.match == 1], \
               [r.data[1] for r in rows if r.match == 1],
    xdn, ydn = [r.data[0] for r in rows if r.match == 0], \
               [r.data[1] for r in rows if r.match == 0]
    plot(xdm, ydm, 'bo')
    plot(xdn, ydn, 'ro')
    show()

# plot_age_matches(ages_only)


def linear_train(rows):

    averages = {}
    counts = {}

    for row in rows:
        cl = row.match
        averages.setdefault(cl, [0.0] * (len(row.data)))
        counts.setdefault(cl, 0)
        for i in range(len(row.data)):
            averages[cl][i] += float(row.data[i])
        counts[cl] += 1

    for cl, avg in averages.items():
        for i in range(len(avg)):
            avg[i] /= counts[cl]

    return averages

def dot_product(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])

def dp_classify(point, avgs):
    b = (dot_product(avgs[1], avgs[1]) - dot_product(avgs[0], avgs[0])) / 2
    y = dot_product(point, avgs[0]) - dot_product(point, avgs[1]) + b
    if y > 0:
        return 0
    else:
        return 1

# avgs = linear_train(ages_only)
# print(dp_classify([30, 30], avgs))

def yesno(v):
    if v == 'yes': return 1
    elif v == 'no': return -1
    else: return 0


def match_count(interest1, interest2):
    l1 = interest1.split(':')
    l2 = interest2.split(':')
    x = 0
    for v in l1:
        if v in l2: x += 1
    return x

def load_numerical():
    oldrows = load_match('matchmaker.csv')
    newrows = []
    for row in oldrows:
        d = row.data
        data = [float(d[0]), yesno(d[1]), yesno(d[2]),
                float(d[5]), yesno(d[6]), yesno(d[7]),
                match_count(d[3], d[8]),
                row.match]
        newrows.append(match_row(data))
    return newrows

dataset = load_numerical()

# ||v1 - v2||
def rbf(v1, v2, gamma=20):
    dv = [v1[i] - v2[i] for i in range(len(v1))]
    l = sum(dv)
    return math.e ** (-gamma * l)

def offset(rows, gamma=10):

    l0 = []
    l1 = []
    for row in rows:
        if row.match == 0:
            l0.append(row.data)
        else:
            l1.append(row.data)

    sum0 = sum(sum([[rbf(v1,v2,gamma)] for v1 in l0 for v2 in l0]))
    sum1 = sum(sum([[rbf(v1,v2,gamma)] for v1 in l1 for v2 in l1]))
    print(sum0, sum1)

    return sum1 / len(l1) ** 2 - sum0 / len(l0) ** 2

print(offset(dataset))

