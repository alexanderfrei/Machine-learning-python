import sys, pickle, time
import pandas as pd


def benchmark(text):
    """
    Benchmark function wrapper 
    :param text: title of block
    :return: print duration of execution, return function result 
    """
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


def pickle_dump(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def save_sub(filename, data1, data2, col1, col2):
    sub = pd.DataFrame()
    sub[col1] = data1
    sub[col2] = data2
    sub.to_csv(filename, index=False)

