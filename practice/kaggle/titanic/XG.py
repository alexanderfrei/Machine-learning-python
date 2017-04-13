
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
    #

    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    ####################################################################################################
    # grid search

    # gbm = XGBClassifier(silent=True)
    # param_grid = {'max_depth': [4],
    #               'learning_rate': [0.1],
    #               'n_estimators': [500],
    #               'subsample': [0.8],
    #               'colsample_bytree': [0.8],
    #               'gamma': [0],
    #               'min_child_weight': [1]
    #               }
    #
    # gs = GridSearchCV(estimator=gbm,
    #                   param_grid=param_grid,
    #                   scoring='accuracy',
    #                   cv=10, n_jobs=-1)
    # gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])
    # print(gs.best_estimator_, gs.best_score_)
    # xbm_best = gs.best_estimator_

    ####################################################################################################
    # fit xgb

    # TODO 1) wrap model fit function 2) parameters tuning 3) early stop

    xbm = XGBClassifier(max_depth=3,
                        learning_rate=0.08,
                        n_estimators=100,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        colsample_bylevel=0.8,
                        gamma=0,
                        reg_alpha=0, reg_lambda=1, min_child_weight=1,
                        objective='binary:logistic')

    import xgboost as xgb
    from sklearn import metrics

    xg_train = xgb.DMatrix(train.iloc[:, 1:], label=train.iloc[:, 0])
    cv_result = xgb.cv(xbm.get_xgb_params(), xg_train,
                       num_boost_round=xbm.get_params()['n_estimators'], nfold=5,
                       metrics='auc', early_stopping_rounds=50)

    xbm.set_params(n_estimators=cv_result.shape[0])
    xbm.fit(train.iloc[:, 1:], train.iloc[:, 0], eval_metric='auc')

    predictions = xbm.predict(train.iloc[:, 1:])
    pred_prob = xbm.predict_proba(train.iloc[:, 1:])[:, 1]

    print("Accuracy : %.4g" % metrics.accuracy_score(train.iloc[:, 0], predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train.iloc[:, 0], pred_prob))

    import matplotlib.pyplot as plt
    feat_imp = pd.Series(xbm.booster().get_fscore()).sort_values(ascending=False)
    # xgb.plot_importance(xbm_best)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    # plt.show()

    xbm_best = xbm

    # ####################################################################################################
    draw_learning_curve(xbm_best, train.iloc[:, 1:], train.iloc[:, 0])

    predictions = xbm_best.predict(test)
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    test = pd.read_csv(os.path.join('data', 'test.csv'))
    predictions = pd.concat((test.iloc[:, 0], predictions), axis=1)
    predictions.to_csv(sub, sep=",", index=False)

