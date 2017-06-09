from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import roc_auc_score
import numpy as np

# Matthews(binary), f1_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(f1_score(y_true, y_pred, average='macro'))  # multi class f1

y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]
print(matthews_corrcoef(y_true, y_pred))  # matthews [-1; 1]
print(f1_score(y_true, y_pred))  # binary f1 [0; 1]


# Gini
def gini(solution, submission):
    """ GINI coefficient for binary classification
    :param solution: vector of predicted
    :param submission: vector of prediction
    :return: Gini
    """

    def gini_vec(list_of_values):
        """ GINI for vector """
        sorted_list = sorted(list_of_values)  # sort vector
        height, area = 0, 0
        for value in sorted_list:
            height += value  # height of curve
            area += height - value / 2.  # area of positive answers under the curve
        fair_area = (height * len(list_of_values)) / 2.  # fair area is a half of rectangle
        return (fair_area - area) / fair_area

    match = np.array(y) == np.array(y_)
    match = match.astype(int)

    return gini_vec(match)

y, y_ = [0, 0, 1, 0, 0], [1, 1, 0, 1, 0]
print(gini(y, y_))
