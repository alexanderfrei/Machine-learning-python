# https://julien.danjou.info/blog/2013/guide-python-static-class-abstract-methods

import math

class Pizza(object):
    def __init__(self, size, cheese, vegetables):
        self.size = size
        self.cheese = cheese
        self.vegetables = vegetables

    def get_size(self):
        return self.size

    @staticmethod
    def mix_ingredients(x, y):
        return x + y

    def cook(self):
        return self.mix_ingredients(self.cheese, self.vegetables)

    radius = 42

    @classmethod
    def get_radius(cls):
        return cls.radius

pizza = Pizza(42, 11, 33)

# метод объекта
m = pizza.get_size
print(m.__self__) # вернет объект

################################################################################################
# статический метод не использует объект класса @ staticmethod
print(pizza.cook())

################################################################################################
# метод класса - используются для наследования @ classmethod
print(pizza.get_radius())


class Pizza(object):
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

    @staticmethod
    def compute_area(radius):
        return math.pi * (radius ** 2)

    @classmethod
    def compute_volume(cls, height, radius):
        return height * cls.compute_area(radius)

    def get_volume(self):
        return self.compute_volume(self.height, self.radius)

pizza = Pizza(42,100)
print(pizza.get_volume())

################################################################################################
# абстрактный метод может не содержать реализации методов,
# но указывает на необходимость их реализации у потомков

# создание абстрактного класса
import abc

class BasePizza(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_ingredients(self):
        """Returns the ingredient list."""

# описание потомка абстрактного класса
class DietPizza(BasePizza):
    @staticmethod
    def get_ingredients():
        return "зелень овощи и прочая трава"

diet = DietPizza()
print(diet.get_ingredients())

base = BasePizza()
print(base.get_ingredients())

################################################################################################
# сочетание декораторов классметод и абстрактметод в абстрактном методе создают абстрактный метод класса
# внутри абстрактного метода, этот же метод у потомков будет совершенно обычным (не методом класса)

class BasePizza(object):
    __metaclass__ = abc.ABCMeta

    default_ingredients = ['cheese']

    @classmethod
    @abc.abstractmethod
    def get_ingredients(cls):
        """Returns the ingredient list."""
        return cls.default_ingredients

# super вызывает родительский класс

class DietPizza(BasePizza):
    def get_ingredients(self):
        return ['egg'] + super(DietPizza, self).get_ingredients()

diet = DietPizza()
print(diet.get_ingredients())

base = BasePizza()
print(base.get_ingredients())
