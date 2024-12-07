import numpy as np
from sklearn.decomposition import PCA
np.random.seed(42)
# Создание искусственных данных
data = np.random.rand(1000, 10)
# Выполнение алгоритма PCA для сокращения размерности до 3-х компонент
pca = PCA(n_components=3)
pca.fit(data)
# Анализ факторной нагрузки
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# Выбор наиболее важных факторов
most_important_factors = np.argsort(np.abs(loadings), axis=0)[::-1][:3]
# Вычисление суммы весов наиболее важных факторов
sum_weights = np.sum(np.abs(loadings[most_important_factors[:, 0], 0]))
# Округление ответа до двух знаков после запятой
sum_weights_rounded = round(sum_weights, 2)
# Вывод результата
print(sum_weights_rounded)