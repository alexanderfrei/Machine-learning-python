
if __name__ == "__main__":

    import os
    from practice.kaggle.titanic.FE import *
    from practice.functions.visual import draw_learning_curve
    sub = 'submission/xgb_cv.csv'

    ####################################################################################################

    train = pd.read_csv(os.path.join('data', 'train.csv'))
    test = pd.read_csv(os.path.join('data', 'test.csv'))

    train, test = names(train, test)
    train, test = age_impute(train, test)
    train, test = cabin_num(train, test)
    train, test = cabin(train, test)
    train, test = embarked_impute(train, test)
    train, test = fam_size(train, test)
    test['Fare'].fillna(train['Fare'].mean(), inplace = True)
    train, test = ticket_grouped(train, test)
    train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                         'Cabin_Letter', 'Name_Title', 'Fam_Size'])
    train, test = drop(train, test)
    replace_cabin(train)
    replace_cabin(test)

    ####################################################################################################
    # fit xgb

    from xgboost import XGBClassifier

    def xgb_fit(alg, X, y, score='auc', cv=5, early_stopping=50):

        import xgboost as xgb
        from sklearn import metrics
        import matplotlib.pyplot as plt

        xg_train = xgb.DMatrix(X, label=y)
        cv_result = xgb.cv(alg.get_xgb_params(), xg_train,
                           num_boost_round=alg.get_params()['n_estimators'], nfold=cv,
                           metrics=score, early_stopping_rounds=early_stopping)

        alg.set_params(n_estimators=cv_result.shape[0])
        alg.fit(X, y, eval_metric=score)

        predictions = alg.predict(X)
        pred_prob = alg.predict_proba(X)[:, 1]

        print(alg)
        print("Accuracy : %.4g" % metrics.accuracy_score(y, predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(y, pred_prob))

        # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        # feat_imp.plot(kind='bar', title='Feature Importances')
        # plt.ylabel('Feature Importance Score')
        # plt.show()

    # 0.81340
    xbm_best = XGBClassifier(max_depth=3,
                             min_child_weight=1,
                             learning_rate=0.01,
                             n_estimators=1000,
                             subsample=0.8,
                             colsample_bytree=0.9)

    xgb_fit(xbm_best, train.iloc[:, 1:], train.iloc[:, 0])

    ####################################################################################################
    # grid search

    # from sklearn.model_selection import GridSearchCV
    #
    # param_grid = {'learning_rate': np.linspace(0.005, 0.03, 10),
    #               }
    #
    # gs = GridSearchCV(estimator=xbm_best,
    #                   param_grid=param_grid,
    #                   scoring='roc_auc',
    #                   cv=5, n_jobs=-1)
    #
    # gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])
    # [print("{}, mean: {}, std: {} ".format(param, score, std)) for (score, std, param) in zip(
    #     gs.cv_results_['mean_test_score'],
    #     gs.cv_results_['std_test_score'],
    #     gs.cv_results_['params'])]
    # print(gs.best_params_, gs.best_score_)
    #
    # xbm_best = gs.best_estimator_

    # ####################################################################################################
    draw_learning_curve(xbm_best, train.iloc[:, 1:], train.iloc[:, 0], cv=5, scoring='roc_auc')

    predictions = xbm_best.predict(test)
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    test = pd.read_csv(os.path.join('data', 'test.csv'))
    predictions = pd.concat((test.iloc[:, 0], predictions), axis=1)
    predictions.to_csv(sub, sep=",", index=False)

