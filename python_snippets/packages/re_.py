'''
мини-гайд по регуляркам

\d Цифры
\D НЕцифры
\w Буквы цифры и _
\W НЕ \w
\s пробельный символ
\S НЕпробельный символ
\b Граница слова
\B НЕграница слова

Немного спецификаторов

| или
? {0,1}
* >=0, жадный
*? >=0, нежадный
+ >=1, жадный
+? >=1, нежадный
exp {m} m включений exp
exp {m,n} от m до n последовательных включений exp, жадный
exp {m,n}? от m до n последовательных включений exp, нежадный
exp1 (?=exp2) exp1, если за ним следует exp2
exp1 (?!exp2) exp1, если за ним НЕ следует exp2
(?<=exp2) exp1 exp1, если перед ним exp2
(?<!exp2) exp1 exp1, если перед ним нет exp2
'''

import re
import string

pattern = re.compile('You')
result = pattern.match('Young Frankenstein')
print(result)

source = 'Young Frankenstein'
m = re.match('You', source)
if m:
    print(m.group())

m = re.match('Frank', source) # match ищет совпадение только в начале источника
if m:
    print(m.group())
else:
    print("no match :(")

m = re.search('Frank', source) # search ищет совпадение в любом месте
if m:
    print(m.start(), m.group())

m = re.match('.*Frank', source)
if m:
    print(m.group())

m = re.findall('n.', source)
print(m, len(m), 'items')

m = re.split('n.', source)
print(m)

m = re.sub('n.','**', source) # замена
print(m)

printable = string.printable
print(re.findall(r'\d',printable))
print(re.findall(r'\b',printable))
print(re.findall(r'\s',printable))
print(re.findall(r'\w',printable))

source = ''' I wish I may, I wish I might
... Have a dish of fish tonight.'''
print(re.findall('I (?=wish)', source))
print(re.findall(r'\b', source)) # неформатированная строка для регулярного выражения

m = re.search(r'(?P<dish>. dish.{3})', source)
print(m.group('dish'))