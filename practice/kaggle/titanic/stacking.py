
if __name__ == "__main__":

    ############################################################################################################

    import pandas as pd

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

    ###########################################################################################################

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
    #
    # ###########################################################################################################
    #

    # ada_params = {
    #     'n_estimators': 340,
    #     'learning_rate': 0.06
    # }
    #
    # et_params = {
    #     'n_estimators': 150,
    #     'max_features': 0.6,
    #     'max_depth': 8,
    #     'min_samples_leaf': 1
    # }
    #
    # gb_params = {
    #     'n_estimators': 400,
    #     'max_features': 0.4,
    #     'max_depth': 9,
    #     'min_samples_leaf': 1
    # }
    #
    # svc_params = {
    #     'kernel': 'linear',
    #     'C': 0.1
    # }
    #
    # rf = RandomForestClassifier(max_features='auto',
    #                                    random_state=2,
    #                                    oob_score=True,
    #                                    n_jobs=-1,
    #                                    criterion='gini',
    #                                    warm_start=True,
    #                                    n_estimators=200,
    #                                    min_samples_split=10,
    #                                    min_samples_leaf=1)
    #
    # dt = DecisionTreeClassifier(max_depth=4)
    # ada = AdaBoostClassifier(**ada_params)
    # et = ExtraTreesClassifier(**et_params)
    # gb = GradientBoostingClassifier(**gb_params)
    # svc = SVC(**svc_params)
    #
    # rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, test)
    # ada_oof_train, ada_oof_test = et_oof(ada, x_train, y_train, test)
    # et_oof_train, et_oof_test = get_oof(et, x_train, y_train, test)
    # gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, test)
    # svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, test)
    # dt_oof_train, dt_oof_test = get_oof(dt, x_train, y_train, test)
    #
    # x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train,
    #                           dt_oof_train), axis=1)
    # x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test,
    #                          dt_oof_test), axis=1)
    #
    # pd.DataFrame(x_train).to_csv('submission/stack_train.csv', index=False)
    # pd.DataFrame(x_test).to_csv('submission/stack_test.csv', index=False)

    ###########################################################################################################

    # x_train = pd.read_csv(os.path.join('submission', 'stack_train.csv'))
    # test = pd.read_csv(os.path.join('submission', 'stack_test.csv'))
    #
    # x_train = x_train.iloc[:,[0,1,2,3,5]]
    # test = test.iloc[:, [0,1,2,3,5]]
    #
    # # corrplot(x_train)
    #
    #
    # gbm_params = {
    #     'learning_rate': [0.01],
    #     'n_estimators': [100],
    #     'max_depth': [3],
    #     'gamma': [0.05],
    #     'subsample': [0.7],
    #     'colsample_bytree': [0.8]
    # }
    #
    # gbm = XGBClassifier(
    #     objective='binary:logistic',
    #     nthread=-1)
    #
    # gbm = grid_search(gbm, gbm_params, x_train, y_train, cv=5)
    # xgb_fit(gbm, x_train, y_train)
    # predictions = gbm.predict(test)
    # # predictions = test.apply(lambda x: round(x)).mean(axis=1).round().astype(int)
    # # draw_learning_curve(gbm, x_train, y_train, higher=1, lower=0.8)
    # sub = 'submission/stack.csv'
    # pd.DataFrame({'PassengerId': pass_id,
    #               'Survived': predictions}).to_csv(sub, index=False)

