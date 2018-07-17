import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


class TargetEncoder(TransformerMixin):
    def __init__(self, group, k=50, f=15):
        self.k = k
        self.f = f
        self.group = group
        self._means = None
        self._prior = None

    def smooth_function(self, n):
        
        return 1 / (1 + np.exp( (n - self.k) / self.f))    
    
    def fit(self, X, y):
        
        group = self.group
        
        # compute prior mean
        prior = y.mean()        
        # concat X and y
        X = pd.concat([X[group], pd.Series(y, name='y', index=X.index)], axis=1)
        # compute mean and base by group
        agg = X.groupby(group)['y'].agg(['mean', 'count'])
        # fit smooth means
        lam = self.smooth_function(agg['count'])
        means = lam * prior + (1 - lam) * agg['mean']

        self._means = means.rename('means').reset_index()
        self._prior = prior
        
        return self

    def transform(self, X):

        assert not self._means is None, "not fitted"
        
        group = self.group
        means = self._means
        prior = self._prior
                
        name = '__'.join(group) + '_mean'
        #print('target encoding:', '__'.join(group))
            
        # map means
        group_means = X[group].merge(how='left', on=group, right=means)['means']
        group_means.index = X.index
        X[name] = group_means.fillna(prior)
        return X