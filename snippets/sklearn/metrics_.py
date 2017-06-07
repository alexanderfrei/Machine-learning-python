from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import roc_auc_score
import numpy as np

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(f1_score(y_true, y_pred, average='macro'))  # multi class f1

y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]
print(matthews_corrcoef(y_true, y_pred))  # matthews [-1; 1]
print(f1_score(y_true, y_pred))  # binary f1 [0; 1]


# Gini



# TODO Gini
# def gini(list_of_values):
#     sorted_list = sorted(list_of_values)
#     height, area = 0, 0
#     for value in sorted_list:
#         height += value
#         area += height - value / 2.
#         print(height, area)
#     fair_area = height * len(list_of_values) / 2.
#     return (fair_area - area) / fair_area
# def gini(solution, submission):
#     """ GINI coef for binary classification
#     G = sum of difference between submission and solution / (2 * n * sum of positive classes in solution )
#     G = 2 * AUC - 1, 2 * roc_auc_score(solution, submission) - 1
#     norm_G = (1 - G) / 2
#     O(n**2)
#     :param solution: vector of predicted
#     :param submission: vector of prediction
#     :return:
#     """
#
#     n = len(solution)
#     df = zip(solution, submission, range(len(solution)))
#     df = sorted(df, key=lambda x: (x[1], -x[2]), reverse=True)  # sort tuples by prediction class and id
#
#     s = 0
#     for i in range(n):
#         for j in range(n):
#             s += np.abs(df[i][0] - df[i][1])  # sum of difference
#
#     s_d = 0
#     for i in df:
#         s_d += i[0]  # sum of positive classes
#
#
#     G = s / (s_d * 2 * n)
#     return G
#
#
