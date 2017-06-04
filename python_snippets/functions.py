""" Briefly python functions guide """

# help(fun) - whole fun code
# fun.__doc__ - first line of docstring


def get_arg(*args, **kwargs):
    """ test function
    :param args: int
    :param kwargs: str = str
    :return:
    """
    try:
        print('1:', args[0])  # 1ый аргумент
        print('Sum:', sum(args))  # arguments' sum
        print('Named arguments: ', kwargs)
        [print(k, end=' ') for k in kwargs.items()]
    except IndexError:
        print('No arguments!')


get_arg(1, 2, 3, 4, red_pill='reality', blue_pill='illusion')
get_arg()


# function is an object like others


def to_print():
    print('successfully get function as argument!')


def run_something(func):  # function func is an argument of other function
    func()


run_something(to_print)


# scope


def dream_hack(dream):
    def level1(arg1):
        print("3..", end='')

        def level2(arg2):
            print("2..", end='')

            def level3(arg3):
                print("1..", end='')

                def level4(arg4):
                    print("spine the top!")
                    print(dream, arg1, arg2, arg3, arg4)  # function level4 see the score of all parents functions

                level4("Limbo")  # but only function level3 could see function level4!

            level3("Eames'")

        level2("Arthur's")

    level1("Yussuf's")
    return None  # PEP8 recommend to return None for readability


dream_hack("`Inception` dream levels: ")  # nested functions execute in the order of creation

try:
    level1("some text")  # wrong call
except NameError:
    print("No function with name level1 in global scope")

"""Closure
1) This is two functions: usual one and nested inside one  
2) The nested function must refer to a value defined in the enclosing function
3) The enclosing function must RETURN the nested function"""


def make_printer(msg):
    def printer():  # nested function
        print(msg)  # value from enclosing function

    return printer  # return nested function


make_printer('Foo!')

"""anonymous functions 
PEP8 not recommends to use lambda """


def edit_story(words, func):
    for word in words:
        print(func(word), end=" ")
    print("")
    return "~__~"


def w_cap(w):
    return w.capitalize() + '!'


stairs = ['thud', 'meow', 'thud', 'hiss']
edit_story(stairs, w_cap)  # preferable way
edit_story(stairs, lambda w: w.capitalize() + '!')


""" Decorators
1) Get function as argument 
2) Return OTHER FUNCTION as result 
functions could have more than one decorator """


# using closure
def function_description(func):  # wrap function func
    """ this decorator print name, arguments of function, and return its result """

    def new_function(*args, **kwargs):  #
        print('Running function:', func.__name__)  # we remember about func function because of enclosing scope
        print('Positional arguments:', args)
        print('Keyword arguments:', kwargs)
        result = func(*args, **kwargs)  # execute function and save result
        print('Result:', result)
        print("End of desc\n")
        return None

    return new_function  # return new function


def add_ints(a, b):
    return a + b

decor = function_description(add_ints)
decor(1, 2)
decor_story = function_description(edit_story)
decor_story(stairs, w_cap)  # OR:


@function_description
def add_ints(a, b):
    return a + b
add_ints(42, 1)


"""better way using functools.wraps 
decorated function is picklable """

from functools import wraps
def function_description(func):  # wrap function func
    """ this decorator print name, arguments of function, and return its result """

    @wraps(func)
    def new_function(*args, **kwargs):
        print('Running function:', func.__name__)
        print('Positional arguments:', args)
        print('Keyword arguments:', kwargs)
        result = func(*args, **kwargs)
        print('Result:', result)
        print("End of desc\n")
        return None

    return new_function


@function_description
def add_ints(a, b):
    return a + b
add_ints(42, 777)

