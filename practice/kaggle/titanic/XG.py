
if __name__ == "__main__":

    ############################################################################################################

    import os
    from practice.kaggle.titanic.FE import *
    from practice.functions.preprocessing import *
    from practice.functions.fit import *
    from practice.functions.visual import *
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier

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

    ############################################################################################################

    # 0.81340
    xbm_best = XGBClassifier(max_depth=3,
                             min_child_weight=1,
                             learning_rate=0.01,
                             n_estimators=1000,
                             subsample=0.8,
                             colsample_bytree=0.9)
    xgb_fit(xbm_best, train.iloc[:, 1:], train.iloc[:, 0])

    # param_grid = {'subsample': np.linspace(0.75, 0.85, 3)}
    # xbm_best = grid_search(xbm_best, param_grid, train.iloc[:, 1:], train.iloc[:, 0])

    # draw_learning_curve(xbm_best, train.iloc[:, 1:], train.iloc[:, 0], cv=5, scoring='roc_auc')

    sub = 'submission/xgb_best.csv'
    save_submission(sub, test, xbm_best, 'Survived', pass_id)

