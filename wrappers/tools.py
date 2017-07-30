
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


def pickle_it(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def depickle_it(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


