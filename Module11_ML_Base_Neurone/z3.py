# Создаем искусственные данные
import numpy as np

np.random.seed(42)

X = np.random.randn(100, 5)  # матрица 100x5

# Определяем архитектуру нейронной сети
input_size = 5
hidden_size = 3
output_size = 1

# Инициализируем веса случайным образом
W1 = np.random.randn(input_size, hidden_size)  # матрица весов 5x3
W2 = np.random.randn(hidden_size, output_size)  # матрица весов 3x1


# Определяем функцию активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Прямой проход по нейронной сети
hidden_layer = sigmoid(np.dot(X, W1))  # скрытый слой
output_layer = np.dot(hidden_layer, W2)  # выходной слой

# Вычисляем среднеквадратичную ошибку
y_true = np.random.randn(100, 1)  # реальные значения
mse = np.mean((output_layer - y_true) ** 2)

print(mse)
