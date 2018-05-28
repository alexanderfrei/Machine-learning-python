def fib_din(n):
    """ Dynamic programming """
    fn = 1
    i = 2
    cache = [1, 1]
    while i < n:
        fn = sum(cache)
        cache = [cache[1], fn]
        i += 1
    return fn


def matrix_pow(x, n, I, mult):
    """
    Matrix x in power n. 
    n = positive integer, I - identity matrix.
    """
    if n == 0:
        return I
    elif n == 1:
        return x
    else:
        y = matrix_pow(x, n // 2, I, mult)
        y = mult(y, y)
        if n % 2:
            y = mult(x, y)
        return y


def matrix_multiply(A, B):
    """ 
    matrix multiply with dims > 1 (not vectors) 
    """
    BT = list(zip(*B))
    return [[sum(a * b for a, b in zip(row_a, col_b))
             for col_b in BT]
            for row_a in A]


def fib_matrix(n):
    F = matrix_pow([[1, 1], [1, 0]], n, [[1, 0], [0, 1]], matrix_multiply)
    return F[0][1]


print(fib_matrix(10000) == fib_din(10000))