import sys
sys.stdin = "4 8\n1 2 6\n1 3 2\n1 4 10\n2 4 4\n3 1 5\n3 2 3\n3 4 8\n4 2 1\n1 4"

# sys.stdin = """4 2
# 3 1 5
# 4 2 1
# 4 1"""
#
sys.stdin = """5 4
1 2 1
2 3 1
3 4 1
4 5 1
1 5"""

sys.stdin = sys.stdin.split("\n")

data = []
for line in sys.stdin:
    data.append(line)

inf = float('inf')

# initialize
size = data[0]
s = data[-1][0] #start vertex
f = data[-1][2] #finish vertex
w = data[1:len(data)-1]
for i in range(len(w)):
    w[i] = w[i].split(' ') #weights
V = []
for i in w:
    if i[0] not in V:
        V.append(i[0])
    if i[1] not in V:
        V.append(i[1]) # all vertex
d = {}
for v in V:
    d[v] = inf # массив кратчайшего пути до вершины
d[s] = 0
Q = V # vertex to drop
q = s # start vertex to check

print(d)
if s in V and f in V: # check data
    while len(Q) > 0:

        cont = False
        for side in w:
            wg = int(side[2])
            to = side[1]
            if not cont:
                if to == f:
                    cont = True
            if side[0] == q:
                    if d[to] > d[q] + wg:
                        d[to] = d[q] + wg

        if not cont: # проверка на входящие ребра в финишную вершину
            break

        # Remove no more needed sides
        for i in w:
            if i[0] == s or i[1] == s:
                w[w.index(i)] = None
        while None in w: w.remove(None)

         # remove vertex

        Q.remove(q)

        # end with reaching finish vertex

        min = -1
        for key,value in d.items():
            if key in Q and (value < min or min == -1):
                key_min = key
                min = value
        q = key_min

    if d[f] != inf:
        print(d[f])
    else:
        print("-1")
else:
    print("-1")
