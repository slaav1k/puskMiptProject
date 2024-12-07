import numpy as np
from sklearn.decomposition import TruncatedSVD


# Создаем матрицу рейтингов пользователей и товаров
R = np.array([[3, 1, 2, 3],
[4, 3, 4, 3],
[2, 2, 1, 5],
[1, 5, 5, 2]])
# Для примера будем считать, что нам нужно предсказать рейтинг для пользователя 2 и товара 4
user_id = 2
item_id = 4
n_components = 2 # количество главных компонент, которые мы оставляем
# Ищем среднее значение рейтингов для каждого товара и вычитаем его из матрицы рейтингов
item_means = np.mean(R, axis=0)
R_norm = R - item_means
# Вычисляем сингулярное разложение матрицы рейтингов
svd = TruncatedSVD(n_components=n_components)
U, s, Vt = svd.fit_transform(R_norm), np.diag(svd.singular_values_), svd.components_
# Определяем размерность матрицы рейтингов и уменьшаем размерность сингулярным разложением
n_users, n_items = R_norm.shape

S = s
U_red = U
Vt_red = Vt
R_pred = np.dot(np.dot(U_red, S), Vt_red) + item_means
# Округляем предсказание до одной десятой
rating_pred = round(R_pred[user_id - 1, item_id - 1] , 1)
print(f"Предсказанный рейтинг для пользователя {user_id} и товара {item_id}: {rating_pred}")