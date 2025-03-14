import math

# Коэффициенты модели
w0 = -5
w1 = 0.002
w2 = 0.5
w3 = 0.1

# Признаки
x1 = 1500  # сумма покупки
x2 = 1      # категория "Электроника"
x3 = 20     # длительность сессии

# Линейная комбинация признаков
z = w0 + w1 * x1 + w2 * x2 + w3 * x3

# Вероятность покупки
probability = 1 / (1 + math.exp(-z))

# Округляем до двух знаков после запятой
rounded_probability = round(probability, 2)
print(rounded_probability)
