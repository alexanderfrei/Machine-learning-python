from multiprocessing.pool import Pool
from functools import partial


def my_fun2(x, general_const):

    return 2*x + general_const


if __name__ == "__main__":

    input_list = [3,4,5]
    my_const = 1000
    num_of_processes = 4
    pool = Pool(num_of_processes)
    result_list = pool.map(partial(my_fun2, general_const=my_const), input_list)
    ## if you prefer, you can also separate them (just another layout, does not change anything)
    #  my_fun2_partial = partial(my_fun2, general_const=my_const)
    #  result_list = pool.map(my_func2_partial, input_list)
    pool.close()
    pool.join()
    print(result_list)
    #  should be [1006,1008,1010]