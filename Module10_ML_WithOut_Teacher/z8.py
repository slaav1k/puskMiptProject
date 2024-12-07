import numpy as np
from numpy.linalg import svd
np.random.seed(42)
# Создание матрицы рейтингов
ratings = np.random.randint(1, 6, size=(4, 4))
# Разложение матрицы с помощью SVD
U, s, Vt = svd(ratings, full_matrices=False)

S = np.diag(s[:2])
U = U[:, :2]
Vt = Vt[:2, 2]

# Вычисление доли общей дисперсии
variance_explained = np.sum(S**2) / np.sum(s**2)
print(round(variance_explained, 2))
