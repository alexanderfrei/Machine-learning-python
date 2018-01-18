# block hungarian optimization


# import
import pandas as pd
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from numba import jit
import datetime as dt


# metric
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
def avg_normalized_happiness(pred, child_pref, gift_pref):
    n_children = 1000000  # n children to give
    n_gift_type = 1000  # n types of gifts available
    n_gift_quantity = 1000  # each type of gifts are limited to this quantity
    n_gift_pref = 100  # number of gifts a child ranks
    n_child_pref = 1000  # number of children a gift ranks
    twins = math.ceil(0.04 * n_children / 2.) * 2  # 4% of all population, rounded to the closest number
    triplets = math.ceil(0.005 * n_children / 3.) * 3  # 0.5% of all population, rounded to the closest number
    ratio_gift_happiness = 2
    ratio_child_happiness = 2

    tmp_dict = np.zeros(n_gift_quantity, dtype=np.uint16)
    for i in np.arange(len(pred)):
        tmp_dict[pred[i][1]] += 1
    for count in np.arange(n_gift_quantity):
        assert count <= n_gift_quantity, "product count > 1000"

    # check if triplets have the same gift
    for t1 in np.arange(0, triplets, 3):
        triplet1 = pred[t1]
        triplet2 = pred[t1 + 1]
        triplet3 = pred[t1 + 2]
        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1], "Triplets error: "

    # check if twins have the same gift
    for t1 in np.arange(triplets, triplets + twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
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

        if np.sum(gift_pref[child_id] == gift_id):
            child_happiness = (n_gift_pref - np.where(gift_pref[child_id] == gift_id)[0]) * ratio_child_happiness
            tmp_child_happiness = child_happiness[0]
        else:
            tmp_child_happiness = -1

        if np.sum(child_pref[gift_id] == child_id):
            gift_happiness = (n_child_pref - np.where(child_pref[gift_id] == child_id)[0]) * ratio_gift_happiness
            tmp_gift_happiness = gift_happiness[0]
        else:
            tmp_gift_happiness = -1

        total_child_happiness += tmp_child_happiness
        total_gift_happiness[gift_id] += tmp_gift_happiness

    # print('normalized child happiness=',
    #       float(total_child_happiness) / (float(n_children) * float(max_child_happiness)),
    #       ', normalized gift happiness', np.mean(total_gift_happiness) / float(max_gift_happiness * n_gift_quantity))

    denominator1 = n_children * max_child_happiness
    denominator2 = n_gift_quantity * max_gift_happiness * n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    return float(math.pow(total_child_happiness * multiplier, 3) + math.pow(np.sum(total_gift_happiness), 3)) / float(
        math.pow(common_denom, 3))


def main(block_size, iter_number, block_number, initial_solution):

    # optimization function
    def optimize_block(child_block, current_gift_ids):

        gift_block = current_gift_ids[child_block]
        C = np.zeros((block_size, block_size))
        for i in range(block_size):
            c = child_block[i]
            for j in range(block_size):
                g = gift_ids[gift_block[j]]
                C[i, j] = child_happiness[c][g]
        row_ind, col_ind = linear_sum_assignment(C)
        return child_block[row_ind], gift_block[col_ind]

    print("run script..")
    # data load
    child_data = pd.read_csv('../input/child_wishlist_v2.csv', header=None).drop(0, 1).values
    gift_data = pd.read_csv('../input/gift_goodkids_v2.csv', header=None).drop(0, 1).values

    n_children = 1000000
    n_gift_type = 1000
    n_gift_quantity = 1000
    n_child_wish = 100
    triplets = 5001
    twins = 40000
    tts = triplets + twins

    # happiness matrix
    print("compute gift HM..")
    gift_happiness = (1. / (2 * n_gift_type)) * np.ones(
        shape=(n_gift_type, n_children), dtype = np.float32)

    for g in range(n_gift_type):
        for i, c in enumerate(gift_data[g]):
            gift_happiness[g,c] = -2. * (n_gift_type - i)

    print("compute child HM..")
    child_happiness = (1. / (2 * n_child_wish)) * np.ones(
        shape=(n_children, n_gift_type), dtype = np.float32)

    for c in range(n_children):
        for i, g in enumerate(child_data[c]):
            child_happiness[c,g] = -2. * (n_child_wish - i)

    gift_ids = np.array([[g] * n_gift_quantity for g in range(n_gift_type)]).flatten()

    # init score
    initial_sub = '../input/{}'.format(initial_solution)
    subm = pd.read_csv(initial_sub)
    subm['gift_rank'] = subm.groupby('GiftId').rank() - 1
    subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']
    subm['gift_id'] = subm['gift_id'].astype(np.int64)
    current_gift_ids = subm['gift_id'].values

    wish = child_data
    gift_init = gift_data
    gift = gift_init.copy()
    answ_org = np.zeros(len(wish), dtype=np.int32)
    answ_org[subm[['ChildId']]] = subm[['GiftId']]
    score_org = avg_normalized_happiness(np.hstack((np.arange(1000000).reshape(-1, 1),
                                                    answ_org.reshape(-1, 1))),
                                         gift, wish)
    print('Initial score: {:.8f}'.format(score_org))

    block_size = block_size
    n_blocks = int((n_children - tts) / block_size)
    children_rmd = 1000000 - 45001 - n_blocks * block_size
    print('block size: {}, num blocks: {}, children reminder: {}'.
          format(block_size, n_blocks, children_rmd))

    # optimization
    answ_iter = np.zeros(len(wish), dtype=np.int32)
    score_best = score_org
    subm_best = subm
    perm_len = iter_number
    block_len = block_number
    for i in range(perm_len):
        print('Current permutation step is: %d' % (i + 1))
        child_blocks = np.split(np.random.permutation
                                (range(tts, n_children - children_rmd)), n_blocks)
        for child_block in child_blocks[:block_len]:
            start_time = dt.datetime.now()
            cids, gids = optimize_block(child_block, current_gift_ids=current_gift_ids)
            current_gift_ids[cids] = gids
            end_time = dt.datetime.now()
            print('Time spent to optimize this block in seconds: {:.2f}'.
                  format((end_time - start_time).total_seconds()))
            ## need evaluation step for every block iteration
            subm['GiftId'] = gift_ids[current_gift_ids]
            answ_iter[subm[['ChildId']]] = subm[['GiftId']]
            score_iter = avg_normalized_happiness(np.hstack((np.arange(1000000).reshape(-1, 1),
                                                             answ_iter.reshape(-1, 1))),
                                                  gift, wish)
            print('Score delta: {:.8f}'.format(score_iter - score_best))
            if score_iter > score_best:
                subm_best['GiftId'] = gift_ids[current_gift_ids]
                score_best = score_iter

        subm_best[['ChildId', 'GiftId']].to_csv('../submission/improved_sub_{}.csv'.format(score_best), index=False)


if __name__=="__main__":
    main(1500, 15, 5, "../scripts/subm_0.935780275207.csv")
