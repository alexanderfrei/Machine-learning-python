# coding: utf-8

import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import hmean, gmean

def harmonic_mean(ls):
    return len(ls)/sum([1/x for x in ls])
def geometric_mean(ls):
    return np.power(np.prod(ls), 1/len(ls))

submissions_path = "./blending/"
all_files = os.listdir(submissions_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f), index_col=0)
        for f in all_files]
concat_df = pd.concat(outs, axis=1)
cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Averaging
concat_df["target"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)
concat_df.drop(cols, axis=1, inplace=True)

print(hmean([1,2,3]) == harmonic_mean(pd.Series([1,2,3])), gmean([1,2,3]) == geometric_mean(pd.Series([1,2,3])))

# harmonic mean
# Apply ranking, normalization and averaging
sub = (concat_df.rank() / concat_df.shape[0]).copy()
sub['target'] = sub.apply(hmean, 1)

# geometric mean
sub = (concat_df.rank() / concat_df.shape[0]).copy()
sub['target'] = sub.apply(gmean, 1)
sub_gmean = sub.target.copy()

# geometric mean - fast way
sub = (concat_df.rank() / concat_df.shape[0]).copy()
sub['target'] = np.exp(np.mean([sub.target_0.apply(np.log),
                                sub.target_1.apply(np.log),
                                sub.target_2.apply(np.log),
                                sub.target_3.apply(np.log),
                                sub.target_4.apply(np.log)], 0))
print(np.all(np.equal(sub_gmean, sub.target)))

# median ranking
sub = (concat_df.rank() / concat_df.shape[0]).copy()
sub['target'] = sub.median(axis=1)

# Write the output
sub.drop(cols, axis=1, inplace=True)
sub.to_csv("./blending/hmean.csv.gz", compression='gzip')
