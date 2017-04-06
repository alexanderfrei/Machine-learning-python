import sys
import pandas as pd
import numpy as np
import re


def wide_cbc_format(cont_list=[], file="data.csv"):

    """

    :param cont_list: лист непрерывных переменных, например, [1], [1,2], [2,5]
    :param file: файл, по дефолту data.csv, в той же папке
    :return: None
    """

    data = pd.read_csv(file, header=0, sep=';')

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for var in data.columns.values:
        if re.match('att.*', var) is None:
            df1 = pd.concat([df1, data[var]], axis=1)
            df2 = pd.concat([df2, data[var]], axis=1)
        else:
            att_num = re.search('(att)(.*)', var).group(2)
            if int(att_num) in cont_list:
                df1 = pd.concat([df1, data[var]], axis=1).rename(columns={var: "x"+att_num})
                df2 = pd.concat([df2, data[var]], axis=1).rename(columns={var: "x"+att_num})
            else:
                var_dummies = pd.get_dummies(data[var], prefix='x'+att_num)
                df1 = pd.concat([df1, var_dummies], axis=1)
                df2 = pd.concat([df2, var_dummies.iloc[:,:-1]], axis=1)

    df1.drop([v for v in df1.columns.values if re.match('.*_0', v)], inplace=True, axis=1)
    df2.drop([v for v in df2.columns.values if re.match('.*_0', v)], inplace=True, axis=1)

    df1.to_csv("data_full.csv", index=False)
    df2.to_csv("data_no_reference.csv", index=False)


wide_cbc_format()
wide_cbc_format([1,3])
wide_cbc_format([1,3], "data.csv")

