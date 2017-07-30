# TODO check * 50000 int64 user x product
# TODO check NAN user x product
# TODO test treshold
# TODO add word2vec segmentation for products

import sys
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import os
import time

# DATASET_PATH = 'C:/datasets/instacart/input'
DATASET_PATH = 'D:/datasets/instacart/input'


def run(text):
    def decor(func):
        def wrapper(*args, **kwargs):
            sys.stdout.write("{}.. ".format(func.__name__))
            sys.stdout.flush()
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            print("{:.1f} sec".format(t2 - t1))
            return res
        return wrapper
    return decor


@run("some text")
def sleep10():
    time.sleep(10)


@run("load data")
def load():

    priors = pd.read_csv('order_products__prior.csv', dtype={
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16,
        'reordered': np.int8})

    train = pd.read_csv('order_products__train.csv', dtype={
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16,
        'reordered': np.int8})

    orders = pd.read_csv('orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

    products = pd.read_csv('products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
                           usecols=['product_id', 'aisle_id', 'department_id'])

    return priors, train, orders, products


@run("feat products")
def feat_products(df):
    tmp_df = pd.DataFrame()
    tmp_df['orders'] = priors.groupby(priors.product_id).size().astype(np.int32) # total
    tmp_df['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32) # reorder
    tmp_df['reorder_rate'] = (tmp_df.reorders / tmp_df.orders).astype(np.float32)
    df = df.join(tmp_df, on='product_id')
    df.set_index('product_id', drop=False, inplace=True)
    del tmp_df
    return df


@run("join orders to priors")
def join_orders_to_priors(od, pr):

    od.set_index('order_id', inplace=True, drop=False)
    pr = pr.join(od, on='order_id', rsuffix='_')
    pr.drop('order_id_', inplace=True, axis=1)
    return pr


@run("user features")
def user_features():

    # from orders
    usr_ord = pd.DataFrame()
    usr_ord['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
    usr_ord['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

    # from priors
    df = pd.DataFrame()
    df['total_items'] = priors.groupby('user_id').size().astype(np.int16)
    df['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
    df['total_distinct_items'] = df.all_products.map(len).astype(np.int16)

    df = df.join(usr_ord)
    df['average_basket'] = (df.total_items / df.nb_orders).astype(np.float32)

    del usr_ord
    return df


@run("feat user x product")
def feat_user_product():

    global priors
    priors['user_product'] = priors.user_id * 50000 + priors.product_id  # max product id < 50000
    d = dict()
    for row in priors.itertuples():
        z = row.user_product
        if z not in d:
            d[z] = (1,
                    (row.order_number, row.order_id),
                    row.add_to_cart_order)
        else:
            d[z] = (d[z][0] + 1,
                    max(d[z][1], (row.order_number, row.order_id)),
                    d[z][2] + row.add_to_cart_order)

    df = pd.DataFrame.from_dict(d, orient='index')
    del d
    
    df.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
    df.nb_orders = df.nb_orders.astype(np.int16)
    df.last_order_id = df.last_order_id.map(lambda x: x[1]).astype(np.int32)
    df.sum_pos_in_cart = df.sum_pos_in_cart.astype(np.int16)
    del priors
    
    return df


@run("train/test split")
def split_orders():

    test_ord = orders[orders.eval_set == 'test']
    train_ord = orders[orders.eval_set == 'train']

    global train
    train.set_index(['order_id', 'product_id'], inplace=True, drop=False)  # set 2-level index
    return test_ord, train_ord


def features(selected_orders, labels_given=False):

    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0

    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]

    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list

    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] = df.user_id.map(users.average_basket)

    print('order related features')
    # df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders

    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id * 50000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(user_x_product.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(user_x_product.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(user_x_product.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(
        lambda x: min(x, 24 - x)).astype(np.int8)
    # df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)

    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return df, labels


def pickle_it(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def depickle_it(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":

    os.chdir(DATASET_PATH)

    # prepare data

    priors, train, orders, products = load()
    products = feat_products(products)
    priors = join_orders_to_priors(orders, priors)
    users = user_features()
    user_x_product = feat_user_product()
    test_orders, train_orders = split_orders()
    df_train, labels = features(train_orders, labels_given=True)

    # dump

    pickle_it(train_orders, 'train_orders.pickle')
    pickle_it(test_orders, 'test_orders.pickle')
    pickle_it(user_x_product, 'user_x_product.pickle')
    pickle_it(users, 'users.pickle')
    pickle_it(products, 'products.pickle')
    pickle_it(train, 'train.pickle')
    pickle_it(labels, 'labels.pickle')
    pickle_it(df_train, 'df_train.pickle')

    # load

    # train = depickle_it('train.pickle')
    # train_orders = depickle_it('train_orders.pickle')
    # user_x_product = depickle_it('user_x_product.pickle')
    # users = depickle_it('users.pickle')
    # products = depickle_it('products.pickle')
    #
    # df_train = depickle_it('df_train.pickle')
    # test_orders = depickle_it('test_orders.pickle')
    # labels = depickle_it('labels.pickle')
    #
    # # train lightgbm
    #
    # f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
    #             'user_average_days_between_orders', 'user_average_basket',
    #             'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
    #             'aisle_id', 'department_id', 'product_orders', 'product_reorders',
    #             'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
    #             'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
    #             'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order'
    #
    # d_train = lgb.Dataset(df_train[f_to_use],
    #                       label=labels,
    #                       categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'
    # del df_train
    #
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': {'binary_logloss'},
    #     'num_leaves': 96,
    #     'max_depth': 10,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.95,
    #     'bagging_freq': 5
    # }
    # ROUNDS = 100
    #
    # bst = lgb.train(params, d_train, ROUNDS)
    # lgb.plot_importance(bst, figsize=(9,20))
    #
    # del d_train
    #
    # # test
    #
    # df_test, _ = features(test_orders)
    #
    # print('light GBM predict')
    # preds = bst.predict(df_test[f_to_use])
    #
    # df_test['pred'] = preds
    #
    # TRESHOLD = 0.22  # guess, should be tuned with crossval on a subset of train data
    #
    # d = dict()
    # for row in df_test.itertuples():
    #     if row.pred > TRESHOLD:
    #         try:
    #             d[row.order_id] += ' ' + str(row.product_id)
    #         except:
    #             d[row.order_id] = str(row.product_id)
    #
    # for order in test_orders.order_id:
    #     if order not in d:
    #         d[order] = 'None'
    #
    # sub = pd.DataFrame.from_dict(d, orient='index')
    #
    # sub.reset_index(inplace=True)
    # sub.columns = ['order_id', 'products']
    # sub.to_csv('sub.csv', index=False)
