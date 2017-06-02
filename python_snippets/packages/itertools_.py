import itertools

for item in itertools.chain([1,2,3], ['a','b','c']):
    print(item, end=' ')
i = 0; print()

for item in itertools.cycle([1,2,3]):
    print(item, end=' ')
    i += 1
    if i == 10:
        break
print()

for item in itertools.accumulate([1,2,3,4]):
    print(item, end=' ')
print()

def multiply(a,b):
    return a * b

for item in itertools.accumulate([1,2,3,4], multiply):
    print(item, end=' ')
