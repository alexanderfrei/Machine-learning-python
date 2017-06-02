# SQLite

import sqlite3

# connection -> cursor
conn = sqlite3.connect('enterprise.db')
curs = conn.cursor()

# SQL query
curs.execute('''
CREATE TABLE zoo (critter VARCHAR(20) PRIMARY KEY, count INT, damages FLOAT)
''')

curs.execute('INSERT INTO zoo VALUES("duck", 5, 0.0)')
curs.execute('INSERT INTO zoo VALUES("bear", 2, 1000.0)')

# заполнитель, маска для защиты от SQL-инъекций

ins = 'INSERT INTO zoo (critter, count, damages) VALUES(?, ?, ?)'
curs.execute(ins, ('weasel', 1, 2000.0))
try:
    curs.execute(ins, ('vvvvv', 2000.0)) # запрос с ошибкой
except:
    print('Неверное число аргументов!')

curs.execute('SELECT * FROM zoo')
rows = curs.fetchall()

curs.execute('SELECT * FROM zoo ORDER BY count')
rows = curs.fetchall()

curs.execute('SELECT * FROM zoo WHERE damages = (SELECT MAX(damages) FROM zoo)')
rows = curs.fetchall()
curs.execute('DROP TABLE zoo')

curs.close()
conn.close()


