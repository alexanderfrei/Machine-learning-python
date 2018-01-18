
"""
lucky ticket

dynamic programming solution:
1_ We have even 2-part ticket of numbers [0;9], with length N, half of length N/2 = n
2_ Range of sum combinations is [0;9*n]: [0;27], {000}, {999} for N=6
3_ Denote number of combinations of one sum = k as D(k,n), f.e. D(0,3)=1: {000}
4_ Then number of tickets for one sum k is D(k,n) * D(k,n) - all possible combinations of numbers rearrangement
5_ So, number of all possible tickets is L = sum(D(k,n)^2) for k in [0;9*n]
6_ We could express D(k,n) recurrently:
    D(k, n) = sum(D(k-j, n-1)) for j in [0;9],
    D(k<0) = 0,
    D(0,0) = 1

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
        array = [sum(array[x::-1]) if x < 10 else sum(array[x:x-10:-1])
                 for x in range(len(array))]
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
print("Recurrent: %fs" % timeit.timeit(wrapped, number=6))

# wrapped = wrapper(happy_tickets_2,6)
# print("Brute force: %fs" % timeit.timeit(wrapped, number=6))
