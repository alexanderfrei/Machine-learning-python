
from collections import defaultdict, OrderedDict

periodic_table = defaultdict(int) # аргумент - функция
periodic_table['Oxygen'] = 8
periodic_table['Lithium']
print(periodic_table)

# OrderedDict

quotes = OrderedDict([
    ('Curly', 'Nyuk nyuk!'),
    ('Moe', 'A wise guy, huh?'),
    ('Larry', 'Ow!')
])

for stooge in quotes:
    print(stooge)
