
""" parallel run with many cpu cores """


from time import time
from concurrent.futures import ProcessPoolExecutor
from functools import wraps

numbers = [(10 ** 7 + 9, 10 ** 10 + 7), (10 ** 7 + 1, 10 ** 10 + 123), (10 ** 7 + 22, 10 ** 10 + 333),
           (10 ** 6 + 5, 10 ** 10 + 7)]

# non parallelled


def benchmark(func, p):

    @wraps(func)
    def wrapper(n):
        start = time()
        if not p:
            results = list(map(func, n))
        else:
            # results = list(map(func, num))
            pool = ProcessPoolExecutor(max_workers=3)
            results = list(pool.map(func, n))  # parallelled version of map()
        end = time()
        print("Execution time: {:.3f} sec, result: {}, parallelled: {}".format(end - start, results, p))

    return wrapper


def gcd(pair):
    a, b = pair
    low = min(a, b)
    for i in range(low, 0, -1):
        if a % i == 0 and b % i == 0:
            return i

gcd1 = benchmark(gcd, False)
gcd2 = benchmark(gcd, True)
gcd1(numbers)
gcd2(numbers)



