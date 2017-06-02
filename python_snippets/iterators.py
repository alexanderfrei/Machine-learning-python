#генератор

#генераторы нужны для создания огромных последовательностей без их полной загрузки в память
#генератор можно создать включением или функцией
#генератор срабатывает полностью лишь однажды

#Каждый раз, когда вы итерируете через генератор, он отслеживает,
# где он находился во время последнего вызова, и возвращает следующее значение.

number_gen = (number for number in range(1,1000)) # круглые () скобки
print(type(number_gen))
for num in number_gen:
    if num> 997: print(num)
number_gen = (number for number in range(1,1000)) # повторяем создание генератора
number_list = list(number_gen)
print(number_list)
number_list = list(number_gen) # пустой лист, т.к. генератор уже сработал раньше
print(number_list)

# определение генератора через функцию (lazy reading)

def my_range(first=0, last=10, step=1):
    number = first
    while number < last:
        yield number
        number += step
ranger = my_range()
print(ranger)
for x in ranger:
    print(x, end=' ')

