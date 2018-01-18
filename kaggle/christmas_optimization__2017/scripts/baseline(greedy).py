
# coding: utf-8


import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from numba import jit



@jit(nopython=True) 
def gcd(x, y):
    while y != 0:
        x, y = y, x % y
    return x

@jit(nopython=True)
def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    return a * b // gcd(a, b)

@jit(nopython=True)
def anh(pred, child_pref, gift_pref):
    
    n_children = 1000000 # n children to give
    n_gift_type = 1000 # n types of gifts available
    n_gift_quantity = 1000 # each type of gifts are limited to this quantity
    n_gift_pref = 100 # number of gifts a child ranks
    n_child_pref = 1000 # number of children a gift ranks
    twins = math.ceil(0.04 * n_children / 2.) * 2    # 4% of all population, rounded to the closest number
    triplets = math.ceil(0.005 * n_children / 3.) * 3    # 0.5% of all population, rounded to the closest number
    ratio_gift_happiness = 2
    ratio_child_happiness = 2    

    tmp_dict = np.zeros(n_gift_quantity, dtype=np.uint16)
    for i in np.arange(len(pred)):
        tmp_dict[pred[i][1]] += 1
    for count in np.arange(n_gift_quantity):
        assert count <= n_gift_quantity, "product count > 1000"
    
    # check if triplets have the same gift
    for t1 in np.arange(0,triplets,3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1], "Triplets error: "
                
    # check if twins have the same gift
    for t1 in np.arange(triplets,triplets+twins,2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        assert twin1[1] == twin2[1], "Twins error"

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)
    
    for i in np.arange(len(pred)):
        row = pred[i]
        child_id = row[0]
        gift_id = row[1]
        
        # check if child_id and gift_id exist
        assert child_id < n_children, "child_id >= n_children"
        assert gift_id < n_gift_type, "gift_id >= n_gift_type"
        assert child_id >= 0, "child_id < 0"
        assert gift_id >= 0, "gift_id < 0"
        
        if np.sum(gift_pref[child_id]==gift_id):
            child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness
            tmp_child_happiness = child_happiness[0]
        else:
            tmp_child_happiness = -1
        
        if np.sum(child_pref[gift_id]==child_id):
            gift_happiness = (n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness
            tmp_gift_happiness = gift_happiness[0]
        else:
            tmp_gift_happiness = -1        

        total_child_happiness += tmp_child_happiness
        total_gift_happiness[gift_id] += tmp_gift_happiness
    
    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) ,         ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))

    denominator1 = n_children*max_child_happiness
    denominator2 = n_gift_quantity*max_gift_happiness*n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    return float(math.pow(total_child_happiness*multiplier,3) + math.pow(np.sum(total_gift_happiness),3)) / float(math.pow(common_denom,3))    



def init_scores(p1, p2):

    scores = np.zeros((1000000, 1000))

    n_ch = scores.shape[0]
    uti = (100 - np.arange(100)) / p1

    for i in tqdm(np.arange(0, n_ch)):
        for j in np.arange(100):
            scores[i, gift_pref[i, j]] += uti[j]
            
    uti = (1000 - np.arange(1000)) / p2
    for i in np.arange(0, scores.shape[1]):
        for j in np.arange(1000):
            scores[child_pref[i, j], i] += uti[j]    
            
    return scores



def greedy(scores):
    
    gifts = np.full((1000),1000)
    sub = []

    # triplets 
    for i in np.arange(0,5001,3):
        best_gifts = np.argsort(np.sum(scores[i:i+3, :], 0)/3)[::-1]
        for g in best_gifts:
            if gifts[g]>0:
                sub.append([i, g])
                sub.append([i+1, g])
                sub.append([i+2, g])
                gifts[g] -= 3
                break
    # twins
    for i in np.arange(5001,45001,2):
        best_gifts = np.argsort(np.sum(scores[i:i+2, :], 0)/2)[::-1]
        for g in best_gifts:
            if gifts[g]>0:
                sub.append([i, g])
                sub.append([i+1, g])
                gifts[g] -= 2
                break    
    
    # all
    for i in tqdm(np.arange(45001,1000000)):
        best_gifts = np.argsort(scores[i, :])[::-1]
        for g in best_gifts:
            if gifts[g]>0:
                sub.append([i, g])
                gifts[g] -= 1
                break            
                
    return sub



# load data 
print("Load data..")
gift_pref = pd.read_csv('../input/child_wishlist_v2.csv',header=None).drop(0, 1).values
child_pref = pd.read_csv('../input/gift_goodkids_v2.csv',header=None).drop(0, 1).values

print("Compute scores..")
scores = init_scores(100, 1000) # parameters = denominators for childs and gifts

print("Set greedy algo..")
sub = greedy(scores) # greedy algo

score = anh(np.array(sub), child_pref, gift_pref)
print("Score: {}".format(str(score)))



# submission
subdf = pd.DataFrame(sub, columns=['ChildId', 'GiftId'])
subdf.to_csv('./greedy_{}.csv'.format(str(score)), index=False)




