
""" Generators
Generators are iterable object which execute once
Within interruption it remembers the last place of iteration
Generators not load whole iterators in memory
"""

number_gen = (number for number in range(1,1000))  # init generator
print(type(number_gen))

for num in number_gen:
    if num> 997:
        print(num, end=' ')

number_list = list(number_gen)
print('\n{}'.format(number_list)) # generator is empty

# define generator with function (lazy reading)


def my_range(first=0, last=10, step=1):
    number = first
    while number < last:
        yield number * 10  # return value 
        number += step

ranger = my_range()
for x in ranger:
    print(x, end=' ')

