import math


# Функция сигмоид
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Задаем коэффициенты a и b
a = 0.5
b = 0.2

# Входные данные
x1 = 1.0
x2 = 0.5

# Линейная комбинация
linear_combination = a * x1 + b * x2

# Применяем функцию активации (сигмоид)
output = sigmoid(linear_combination)

# Выводим результат
print(output)
