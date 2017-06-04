
""" coroutines """


def my_coroutine():
    while True:
        received = yield
        # do stuff:
        print('Get: ', received)


it = my_coroutine()
next(it)  # start call is necessary to prepare generator
for i in range(5):
    it.send(i)


# TODO:
""" modern python 3.5 way 
https://habrahabr.ru/post/266743/
https://docs.python.org/3/library/asyncio-task.html
https://www.python.org/dev/peps/pep-0492/
"""