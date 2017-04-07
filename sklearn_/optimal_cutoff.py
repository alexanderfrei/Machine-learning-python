import pandas as pd
from sklearn.metrics import roc_curve

def optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

# cutoff
# pred_proba = pipe_lr.predict_proba(X_test)
# cutoff = optimal_cutoff(y_test, pred_proba[:, 1])
# print(pred_proba, y_pred, cutoff)
# print(np.where(pred_proba[:, 1] > cutoff, 1, 0))

