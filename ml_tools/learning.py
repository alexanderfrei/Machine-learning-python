import numpy as np
import pandas as pd
from functools import partial


def cross_val(X, y, model, kf, preprocessing, metric, verbose=False):

    """
    cross-validation with custom preprocessing 
    
    :param X: pandas/numpy training set  
    :param y: pandas/numpy labels 
    :param model: estimator
    :param kf: cross-validation strategy 
    :param preprocessing: custom preprocessing using train, test and labels sets
    :param metric: sklearn-like metric  
    :return: list of cv scores 
    """

    X, y = np.array(X), np.array(y).reshape(-1)
    cv_scores = []

    for i, (train_index, val_index) in enumerate(kf.split(X, y)):

        print("Fold ", i)

        y_train, y_val = y[train_index].copy(), y[val_index].copy()
        X_train, X_val = X[train_index, :].copy(), X[val_index, :].copy()

        X_train, X_val = preprocessing(X_train, X_val, y_train)
        if verbose: print(X_train.shape)

        fit_model = model.fit(X_train, y_train)
        pred = fit_model.predict_proba(X_val)[:, 1]

        cv_scores.append(metric(y_val, pred))

    return cv_scores


def learning_curve_cv(X, y, model, transform, train_size, metric, kf, cv=3):

    """
    Compute train and validation scores for learning curves
    
    :param X: train set  
    :param y: labels
    :param model: estimator 
    :param transform: preprocessing function 
    :param train_size: list of numbers <= train.shape[0] 
    :param metric: sklearn-like metric 
    :param kf: cross-validation strategy (KFold/StratifiedKFold etc.)
    :param cv: number of folds 
    :return: 2 numpy arrays in tuple (train + validation scores)
    """

    transform = partial(transform, feature_names=list(X.columns))
    train_size = np.int32(train_size)
    kf = kf(cv, random_state=42, shuffle=True)

    train_scores = np.zeros((len(train_size), cv), dtype=np.float32)
    valid_scores = np.zeros((len(train_size), cv), dtype=np.float32)

    X, y = np.array(X), np.array(y).reshape(-1)

    for j, num in enumerate(train_size):
        print(j, num)
        for i, (train_index, val_index) in enumerate(kf.split(X[:num, :], y[:num])):
            print("Fold ", i)

            y_train, y_val = y[train_index].copy(), y[val_index].copy()
            X_train, X_val = X[train_index, :].copy(), X[val_index, :].copy()

            X_train, X_val = transform(X_train, X_val, y_train)
            fit_model = model.fit(X_train, y_train)

            train_proba = fit_model.predict_proba(X_train)[:, 1]
            val_proba = fit_model.predict_proba(X_val)[:, 1]

            train_scores[j, i] = metric(y_train, train_proba)
            valid_scores[j, i] = metric(y_val, val_proba)

    return train_scores, valid_scores


def cutoff_metrics(y_valid, predict_proba, cutoff):

    """
    Print table of metrics by probabilities cutoff for bunary classification  
    
    :param y_valid: pandas/numpy vector of true labels 
    :param predict_proba: pandas/numpy vector of prediction probabilities
    :param cutoff: list of cuts
    :return: none
    """

    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
    for cut in cutoff:
        print("f1: {:.3f}\trecall: {:.3f}\tprecision: {:.3f}\tacc: {:.3f}\tcutoff: {:.3f} ".format(
            f1_score(y_valid, predict_proba > cut),
            recall_score(y_valid, predict_proba > cut),
            precision_score(y_valid, predict_proba > cut),
            accuracy_score(y_valid, predict_proba>cut),
            cut
        ))


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,  # Revised to encode validation series
                  val_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series,
                                                                                                   noise_level)


def cv_learn(X, y, X_test, model, kf, metric, xgb_metric, early_stopping_rounds,
             opt_rounds=False, verbose=True, t_encode=True,
             min_samples_leaf=200, smoothing=10, noise_level=0):  # smoothing parameters

    """
    Learning with cross-validation and tearget encoding
    Suppose category variables end with "_cat"
    
    :param X: train, pandas df
    :param y: labels, pandas df
    :param X_test: test, pandas df 
    :param model: xgboost estimator
    :param opt_rounds: is necessary opt_rounds in xgboost?, Bool 
    :param early_stopping_rounds: number of early stopping rounds 
    :param kf: cv strategy 
    :param verbose: Bool
    :param t_encode: is necessary target encoding?, Bool
    :param min_samples_leaf: 
    :param smoothing: 
    :param noise_level: 
    :param metric: metric (probabilities)
    :param xgb_metric: xgb-like metric
    :return: 
    """
    y_valid_pred = 0 * y
    y_test_pred = 0
    f_cats = [c for c in X.columns if '_cat' in c]
    cv_scores = []

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        # Create data for this fold
        y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()
        X_test_cv = X_test.copy()
        if verbose: print("\nFold ", i)

        if t_encode:
            for f in f_cats:
                X_train[f + "_avg"], X_valid[f + "_avg"], X_test_cv[f + "_avg"] = target_encode(
                    trn_series=X_train[f],
                    val_series=X_valid[f],
                    tst_series=X_test_cv[f],
                    target=y_train,
                    min_samples_leaf=min_samples_leaf,
                    smoothing=smoothing,
                    noise_level=noise_level
                )
        # Run model for this fold
        if opt_rounds:
            eval_set = [(X_valid, y_valid)]
            fit_model = model.fit(X_train, y_train,
                                  eval_set=eval_set,
                                  eval_metric=xgb_metric,
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose=False
                                  )
            if verbose:
                print("Best N trees = ", model.best_ntree_limit)
                #                 print( "  Best gini = ", -model.best_score )
        else:
            fit_model = model.fit(X_train, y_train)

        # Generate validation predictions for this fold
        pred = fit_model.predict_proba(X_valid)[:, 1]
        cv_scores.append(metric(y_valid, pred))
        if verbose: print("Gini = ", cv_scores[-1])

        y_valid_pred.iloc[test_index] = pred

        # Accumulate test set predictions
        y_test_pred += fit_model.predict_proba(X_test_cv)[:, 1]

        # clean memory
        del X_test_cv, X_train, X_valid, y_train

    y_test_pred /= kf.n_splits  # Average test set predictions
    m = np.mean(cv_scores)  # mean

    if verbose:
        sd = np.std(cv_scores)  # std
        print("Cv scores:")
        print("{:.4} +- {:.4}".format(m, sd))

    return y_valid_pred, y_test_pred, m