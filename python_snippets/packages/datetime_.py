import time
from datetime import date
from datetime import timedelta
from datetime import datetime

print(time.time()) # epoch
print(time.ctime())

print(datetime.now()) # текущая дата
print(datetime.utcnow())
convert_date = [int(x) for x in str(datetime.now().date()).split('-')] # перевод datetime в date

dt = date(*convert_date)
print(dt)
dt_10000 = dt + timedelta(days = 10000) # сдвиг даты на 10000 дней
print(dt_10000)

print(dt.year)
print(dt.month)
print(dt.day)
print(dt.isocalendar())
print(dt.isoformat())
print(dt.isoweekday())

