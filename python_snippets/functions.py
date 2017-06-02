# TODO read PEP0008
# http://legacy.python.org/dev/peps/pep-0008/

# Функции

# help(функция) - вывод функции целиком
# функция.__doc__ - вывод первой строки описания функции

def get_arg(*args, **kwargs):
    """
    функция теста аргументов python
    :param args: int
    :param kwargs: str = str
    :return:
    """
    try:
        print('1:', args[0])  # 1ый аргумент
        print('Сумма:', sum(args))  # сумма аргументов
        print('Именованные аргументы: ', kwargs)
        [print(k) for k in kwargs.items()]
    except:
        pass

get_arg(1, 2, 3, 4, red_pill='reality', blue_pill='illusion')

# функцию можно передать в качестве аргумента

def to_print():
    print('successfully get function as argument!')

def run_something(func):
    func()

run_something(to_print)

# внутренние функции

def dream(dream):
    def level1(arg1):
        print("3..", end='')
        def level2(arg2):
            print("2..", end='')
            def level3(arg3):
                print("1..", end='')
                def level4(arg4):
                    print("spine the top!")
                    print(dream,arg1,arg2,arg3,arg4)
                    # функция level4 видит все родительские функции и их аргументы
                level4("Limbo") # функцию level4 видит только level3
            level3("Eames'")
        level2("Arthur's")
    level1("Yusuf's")

dream("`Inception` dream levels: ") # внутренние функции выполняются в порядке создания
dream_var = dream("`Inception` dream levels: ")
# level1("bla") - внутренних функций нет в глобальном пространстве переменных
# выполнение кода выше выдаст ошибку

# замыкания

def knights(saying):
    def inner():
        return "We are the knights who say: '%s'" % saying
    return inner

knights("kek")
a = knights("(_(")
b = knights("x_x")
print(a, b)
print(a(),b())

# анонимные функции
# pep8 не рекомендует использование анонимных функций

def edit_story(words, func):
    for word in words:
        print(func(word))
    return "~__~"
stairs = ['thud', 'meow', 'thud', 'hiss']
edit_story(stairs, lambda word: word.capitalize() + '!')

# декораторы
# декоратор — это функция, которая принимает одну функцию и возвращает другую
# часто используется для отладки
# функция может иметь более одного декоратора
# первым выполнится тот, что ближе к функции

# декоратор, возращающий имя функции и аргументы:
def decor_example(func):
    def new_function(*args, **kwargs):
        print('Running function:', func.__name__)
        print('Positional arguments:', args)
        print('Keyword arguments:', kwargs)
        result = func(*args, **kwargs)
        print('Result:', result)
        return result # возвращаем результат функции
    return new_function

decor_func = decor_example(edit_story)
decor_func(stairs, lambda word: word.capitalize() + '!')

@decor_example
def add_ints(a, b):
    return a + b
add_ints(1,2)

