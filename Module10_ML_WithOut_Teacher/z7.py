import numpy as np
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error
np.random.seed(42)

# Создание матрицы рейтингов
ratings = np.random.randint(1, 6, size=(5, 5)).astype(float)

# Разложение матрицы с помощью SVD
U, S, Vt = svds(ratings, k=2)

S_diag = np.diag(S)

# Получение приближенной матрицы рейтингов
ratings_approx = np.dot(np.dot(U, S_diag), Vt)

# Вычисление RMSE
rmse = np.sqrt(mean_squared_error(ratings, ratings_approx))

print(round(rmse, 2))