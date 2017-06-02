# TODO сделать нормальные описания функций

"""
lucky ticket
задача о счастливом билете
"""

import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def happy_tickets(num):
    """
    recurrent solution
    """
    if num % 2 != 0 or num <= 0:
        return '# Error: invalid number'
    array = [1] * 10 + [0] * (num // 2 * 9 - 9)
    for i in range(num // 2 - 1):
        array = [sum(array[x::-1]) if x < 10
                 else sum(array[x:x-10:-1])
                 for x in range(len(array)) ]
    return sum([x**2 for x in array])


def happy_tickets_2(num):
    """
    ineffective brute force solution
    """
    if num % 2 != 0 or num <= 0:
        return '# Error: invalid number'
    eq = 0
    for i in range(10**num):
        sum_l, sum_r = 0, 0
        for j in range(num // 2, num):
            sum_l += i // 10 ** j % 10
        for j in range(num // 2):
            sum_r += i // 10 ** j % 10
        if sum_l == sum_r:
            eq += 1
    return eq

wrapped = wrapper(happy_tickets,6)
print("Reccurent: %fs" % timeit.timeit(wrapped, number=6))

wrapped = wrapper(happy_tickets_2,6)
print("Brute force: %fs" % timeit.timeit(wrapped, number=6))
