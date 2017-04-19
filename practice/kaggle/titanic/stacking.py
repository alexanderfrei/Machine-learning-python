
if __name__ == "__main__":

    ############################################################################################################

    import os
    from practice.kaggle.titanic.FE import *
    from practice.functions.preprocessing import *
    from practice.functions.fit import *
    from practice.functions.visual import *
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier

    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
        ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    pd.set_option('display.width', 200)

    ############################################################################################################

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
    train, test = dummies(train, test, columns=['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                'Cabin_Letter', 'Name_Title', 'Fam_Size'])
    pass_id = test['PassengerId']
    train, test = drop(train, test)
    replace_cabin(train); replace_cabin(test)

    x_train = train.iloc[:, 1:].as_matrix()
    y_train = train.iloc[:, 0].as_matrix()
    test = test.as_matrix()

    ############################################################################################################

    rf_params = {'criterion': 'gini',
                 'n_estimators': 700,
                 'min_samples_split': 5,
                 'min_samples_leaf': 1,
                 'max_features': 'auto',
                 'oob_score': True,
                 'n_jobs': -1}

    n_iter = 15
    pred = np.zeros([test.shape[0], n_iter])
    for i in range(n_iter):
        rf = RandomForestClassifier(**rf_params, random_state=None)
        rf.fit(x_train, y_train)
        print("%.4f" % rf.oob_score_)
        pred[:, i] = rf.predict(test)

    # print(pred.mean(axis=1).round())
    predictions = pred.mean(axis=1).round().astype(int)

    # predictions = rf.predict(test)
    # print("%.4f" % rf.oob_score_)

    # ada_params = {
    #     'n_estimators': 500,
    #     'learning_rate': 0.75
    # }
    #
    # et_params = {
    #     'n_jobs': -1,
    #     'n_estimators': 500,
    #     # 'max_features': 0.5,
    #     'max_depth': 8,
    #     'min_samples_leaf': 2,
    #     'verbose': 0
    # }
    #
    # gb_params = {
    #     'n_estimators': 500,
    #     # 'max_features': 0.2,
    #     'max_depth': 5,
    #     'min_samples_leaf': 2,
    #     'verbose': 0
    # }
    #
    # svc_params = {
    #     'kernel': 'linear',
    #     'C': 0.025
    # }
    #
    # rf = RandomForestClassifier(**rf_params)
    # ada = AdaBoostClassifier(**ada_params)
    # et = ExtraTreesClassifier(**et_params)
    # gb = GradientBoostingClassifier(**gb_params)
    # svc = SVC(**svc_params)
    #
    # rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, test)
    # ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, test)
    # et_oof_train, et_oof_test = get_oof(et, x_train, y_train, test)
    # gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, test)
    # svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, test)
    #
    # x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
    # x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, svc_oof_test), axis=1)

    # gbm = XGBClassifier(
    #     # learning_rate = 0.02,
    #     n_estimators=2000,
    #     max_depth=4,
    #     min_child_weight=2,
    #     # gamma=1,
    #     gamma=0.9,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     objective='binary:logistic',
    #     nthread=-1,
    #     scale_pos_weight=1).fit(x_train, y_train)
    #
    # predictions = gbm.predict(x_test)
    #
    sub = 'submission/rf.csv'
    pd.DataFrame({'PassengerId': pass_id,
                  'Survived': predictions}).to_csv(sub, index=False)

# TODO 0) tune 1-level clf
# TODO 1) add xgboost to first level 2) predictions corrplot (влияет - не влияет?) 3) cv+gridsearch tuning level 2



