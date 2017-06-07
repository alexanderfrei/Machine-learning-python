
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

