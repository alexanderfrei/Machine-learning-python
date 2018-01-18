from numba import jit
import numpy as np
import pandas as pd

#########################################################################################################
# Gini

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    """
    Compute Gini for XGBoost learning 
    :param preds: predictions 
    :param dtrain: xgboost-like vector of labels 
    :return: Gini score, float
    """
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def ginic(actual, pred):
    """
    Fast Gini computation 
    
    :param actual: labels
    :param pred: prediction
    :return: Gini score, float 
    """
    actual = np.asarray(actual)
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2
    return 2 * giniSum / n

def gini_normalizedc(a, p):
    """
    Normalized Gini metric 
    
    :param a: vector of labels  
    :param p: vector of predictions 
    :return: norm. Gini metric, float
    """
    if p.ndim == 2:  # Required for sklearn wrapper
        p = p[:, 1]  # If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)

def gini_scorer(clf, X, y):
    """
    Gini scorer for sklearn  
    
    :param clf: estimator 
    :param X: train set 
    :param y: labels 
    :return: Gini metric, float
    """
    y_proba = clf.predict_proba(X)[:, 1]
    return ginic(y, y_proba) / ginic(y, y)

#########################################################################################################