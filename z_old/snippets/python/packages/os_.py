# запись вывода в файл
fout = open('oops.txt', 'wt')
print('Oops, I created a file.', file= fout)
fout.close()

# os.path.exists
import os
print(os.path.exists('oops.txt'))
print(os.path.exists('./oops.txt'))
print(os.path.exists('.'))
print(os.path.exists('..')) # родительская папка

# проверка типа файла
name = 'oops.txt'
print(os.path.isfile(name))
print(os.path.isdir(name))
print(os.path.isabs('/')) # проверка абсолютного пути на правильность
print(os.path.isabs('incorrect/name'))

# операции
import shutil
shutil.copy('oops.txt','oops_copy.txt')
shutil.move('oops_copy.txt','oops_copy2.txt') # перемещение файла
os.rename('oops_copy2.txt','oops_copy3.txt')
os.remove('oops_copy3.txt')
print(os.path.abspath(name)) # return abspath

# каталоги
if os.path.exists('test'):
    os.rmdir('test')
os.mkdir('test')
print(os.listdir('..'))
print(os.listdir('.'))
os.chdir('test')
print(os.listdir('.'))
