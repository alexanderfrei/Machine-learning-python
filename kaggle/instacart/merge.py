import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import os
import time

# DATASET_PATH = 'C:/datasets/instacart/'
DATASET_PATH = 'D:/datasets/instacart/input'


def run(text):
    def decor(func):
        def wrapper(*args, **kwargs):
            print("{}.. ".format(func.__name__), end="")
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            print("{:.1f} sec".format(t2 - t1))
            return res
        return wrapper
    return decor


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
    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
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


if __name__ == "__main__":

    os.chdir(DATASET_PATH)

    priors, train, orders, products = load()
    products = feat_products(products)
    priors = join_orders_to_priors(orders, priors)
    users = user_features()
    user_x_product = feat_user_product()
    test_orders, train_orders = split_orders()

    df_train, labels = features(train_orders, labels_given=True)

    with open('labels.pickle') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
    with open('merged_train.pickle') as f:
        pickle.dump(df_train, f, pickle.HIGHEST_PROTOCOL)

    df_train.to_csv('merged_train.csv', index=False)

