import numpy as np


# Функция плотности вероятности
def func(x):
    if 0 <= x < 0.5:
        return 0.4 * (x - 1) ** 3 + 0.4
    elif 0.5 <= x < 1.5:
        return 0.3 * x + 0.2
    elif 1.5 <= x < 2:
        return 0.4 * (x - 1) ** 3 + 0.6
    return 0


# Метод численного интегрирования методом трапеций
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    sum_val = (f(a) + f(b)) / 2.0

    for i in range(1, n):
        sum_val += f(a + i * h)

    return sum_val * h


# Теоретическое математическое ожидание
def calculate_mat_expectation(n):
    a, b = 0, 2  # Интервал [0, 2]
    f = lambda x: x * func(x)  # x * f(x)
    return trapezoidal_rule(f, a, b, n)


# Теоретическая дисперсия
def calculate_variance(Mx, n):
    a, b = 0, 2  # Интервал [0, 2]
    f = lambda x: (x - Mx) ** 2 * func(x)  # (x - Mx)^2 * f(x)
    return trapezoidal_rule(f, a, b, n)


# Основной блок программы
if __name__ == "__main__":
    n = 10000  # Число подынтервалов для интегрирования

    # Вычисление математического ожидания
    Mx = calculate_mat_expectation(n)
    print(f"Теоретическое математическое ожидание: {Mx}")

    # Вычисление дисперсии
    Dx = calculate_variance(Mx, n)
    print(f"Теоретическая дисперсия: {Dx}")
