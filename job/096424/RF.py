
if __name__ == "__main__":

    from practice.functions.preprocessing import *
    from practice.functions.fit import *

    df = pd.read_csv('for_rf.csv')
    df = df.iloc[:, 1:]
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    ####################################################################################################
    # RF

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(criterion='gini', n_jobs=-1, oob_score=True, max_features='auto',
                                n_estimators=500,
                                min_samples_split=10)

    # rf = grid_search(rf, param_grid, X, y)

    rf.fit(X, y)
    print("%.4f" % rf.oob_score_)

    print(pd.concat((pd.DataFrame(X.columns, columns = ['variable']),
               pd.DataFrame(rf.feature_importances_, columns = ['importance'])),
              axis = 1).sort_values(by='importance', ascending = False)
          )
