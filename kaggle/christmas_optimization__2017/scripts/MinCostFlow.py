
# coding: utf-8


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ortools.graph import pywrapgraph # https://developers.google.com/optimization/flow/mincostflow



# Load data
kids = pd.read_csv('../input/child_wishlist_v2.csv', header=None).as_matrix()[:, 1:]
print("Kids loaded with shape:")
print(kids.shape)

gifts = pd.read_csv('../input/gift_goodkids_v2.csv', header=None).as_matrix()[:, 1:]
print("Gifts loaded with shape:")
print(gifts.shape)



# Set up global params
nkids = 1000000
nwishes = 100
ngift_types = 1000
neach_gift = 1000
ntriplets = 5001
ntwins = 45001

# Set up global solution variables
gcount = np.zeros(ngift_types, dtype=np.int32)
gsoln = np.zeros(nkids, dtype=np.int32)

# Set up ortools parameters and the all-important arc cost functions
wmax = 75 #nwishes  #Ideally set to nwishes (ie. 100) but needs more memory than I have
ggcost = 1 #cost of arc to generic gift

# https://www.kaggle.com/c/santa-gift-matching/discussion/46559
# mf is used to divide triplet scores by 3 (mf=2) and twins by 2 (mf=3); single kids have mf=6
def wishcost(w, mf):
    return  int(mf * (-2 * (nwishes - w)))

def prefcost(p):
    return  int(-2 * (neach_gift - p) / 1000)



# Set up min max graph
snodes = [] 
enodes = [] 
capacities = []
ucosts = []
supplies = []

for g in range(ngift_types): # 1000
    snodes.append(g) # id
    enodes.append(ngift_types) # 1000
    capacities.append(neach_gift) # 1000
    ucosts.append(0) 
    supplies.append(neach_gift) # 1000
supplies.append(0)

kbase = ngift_types + 1 # 1001
for k in range(nkids): # 1M
    snodes.append(ngift_types)
    enodes.append(kbase + k)
    capacities.append(1)
    ucosts.append(ggcost)
    supplies.append(-1)

for k in range(nkids):
    if k % 250000 == 0:
        print(k)
    for w in range(wmax): #(nwishes):
        snodes.append(int(kids[k, w]))
        enodes.append(int(kbase + k))
        capacities.append(1)
print('nodes and  arcs set up. calculating costs ...')

# costs for bipartite graph
for k in range(ntriplets):
    tb = k // 3
    for w in range(wmax):
        g = kids[k, w]
        kc = wishcost(w, 2)
        for j in range(3):
            kj = 3 * tb + j
            gp = np.where(gifts[g] == kj)[0]
            if gp:
                kc += prefcost(gp)
            if kj != k:
                js = np.where(kids[kj] == g)[0]
                if js:
                    kc += wishcost(js, 2)
        ucosts.append(int(kc))

for k in range(ntriplets, ntwins):
    tb = (k + 1) // 2
    for w in range(wmax):
        g = kids[k, w]
        kc = wishcost(w, 3)
        for j in range(2):
            kj = 2 * tb - j
            gp = np.where(gifts[g] == kj)[0]
            if gp:
                kc += prefcost(gp)
            if kj != k:
                js = np.where(kids[kj] == g)[0]
                if js:
                    kc += wishcost(js, 3)
        ucosts.append(int(kc))

for k in range(ntwins, nkids):
    if k % 100000 == 0:
        print(k)
    for w in range(wmax):
        g = kids[k, w]
        gp = np.where(gifts[g] == k)[0]
        kc = wishcost(w, 6)
        if gp:
            kc += prefcost(gp)
        ucosts.append(int(kc))

print('costs complete')



smcf = pywrapgraph.SimpleMinCostFlow()

# Add each arc
for i in range(0, len(snodes)):
    if i % 10000000 == 0:
        print(i)
    smcf.AddArcWithCapacityAndUnitCost(snodes[i], enodes[i], capacities[i], ucosts[i])

# Add node supplies
for i in range(0, len(supplies)):
    smcf.SetNodeSupply(i, supplies[i])

print('graph instantiated')



st = smcf.SolveMaxFlowWithMinCost()
print('Solved with status', st, ' Pull results....')
print('Maximum flow:', smcf.MaximumFlow(), ' = ', nkids)



# To run a new iteration of tidy-up in interactive mode, start here to skip time-consuming set-up

gsoln[:] = -1
gcount[:] = neach_gift

# Running counts of allocns
trc = 0 # triplets
twc = 0 # twins
skc = 0 # single kids
ukc = 0 # unlucky kids



# Translate graph solution to problem
for i in range(smcf.NumArcs()):
    if i % 5000000 == 0:
        print(i)
    cost = smcf.Flow(i) * smcf.UnitCost(i)
    if cost != 0:
        k = smcf.Head(i) - ngift_types - 1
        g = smcf.Tail(i)
        if g == ngift_types:
            continue
        if k < ntriplets:
            if gsoln[k] == -1:
                tb = k // 3
                for j in range(3):
                    gsoln[tb * 3 + j] = g
                gcount[g] -= 3
                trc += 3
        else:
            if k < ntwins:
                if gsoln[k] == -1:
                    tb = (k + 1) // 2
                    for j in range(2):
                        gsoln[tb * 2 - j] = g
                    gcount[g] -= 2
                    twc += 2
            else:
                if gsoln[k] == -1:
                    gsoln[k] = g
                    gcount[g] -= 1
                    skc += 1

print(trc, twc, skc, ukc, trc+twc+skc+ukc, gcount.min(), gcount.max())



# Fix missing triplets/twins
for k in range(0, ntriplets, 3):
    if gsoln[k] > -1:
        continue
    g = np.argmax(gcount)
    for j in range(3):
        gsoln[k + j] = g
    gcount[g] -= 3
    trc += 3

for k in range(ntriplets, ntwins, 2):
    if gsoln[k] > -1:
        continue
    g = np.argmax(gcount)
    for j in range(2):
        gsoln[k + j] = g
    gcount[g] -= 2
    twc += 2

print(trc, twc, skc, ukc, trc+twc+skc+ukc, gcount.min(), gcount.max())



# Correct for overallocation thanks to tr/tw
# For each negative gcount, steal from a single kid and leave fix to next part
for g in range(ngift_types):
    gc = gcount[g]
    if gc >= 0:
        continue
    kg_soln = np.where(gsoln == g)[0]  
    i = -1
    while gc < 0:
        i += 1
        kv = kg_soln[i]
        if kv < ntwins:
            continue
        gsoln[kv] = -1
        skc -= 1
        gcount[g] += 1
        gc += 1

print(trc, twc, skc, ukc, trc+twc+skc+ukc, gcount.min(), gcount.max())



# Allocate giftless kids
for k in range(ntwins, nkids):
    if gsoln[k] == -1:
        g = np.argmax(gcount)
        gsoln[k] = g
        gcount[g] -= 1
        ukc += 1

print(trc, twc, skc, ukc, trc+twc+skc+ukc, gcount.min(), gcount.max())



# Count total scores (Kaggle metric)
total_child_happiness = 0
total_gift_happiness = np.zeros(ngift_types)

for k in range(nkids):
    gift_id = gsoln[k]
    child_happiness = (nwishes - np.where(kids[k]==gift_id)[0]) * 2
    if not child_happiness:
        child_happiness = -1

    gift_happiness = (neach_gift - np.where(gifts[gift_id]==k)[0]) * 2
    if not gift_happiness:
        gift_happiness = -1

    total_child_happiness += child_happiness
    total_gift_happiness[gift_id] += gift_happiness

nch = float(total_child_happiness)/200000000
ngh = float(np.mean(total_gift_happiness))/2000000
print('normalized child happiness=', nch)
print('normalized gift happiness=', ngh)
print('Estimated Kaggle score', nch ** 3)



# Write solution file
out = open('submit_verGS.csv', 'w')
out.write('ChildId,GiftId\n')
for i in range(nkids):
    out.write(str(i) + ',' + str(gsoln[i]) + '\n')
out.close()

print("Done!")




