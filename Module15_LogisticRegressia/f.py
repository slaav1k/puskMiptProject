import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv("student.csv")

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(columns=["grade"])  # Убираем целевую переменную
y = data["grade"]

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание пайплайна
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=4)),  # Полиномизация до 4 степени
    ('scaler', StandardScaler()),  # Масштабирование данных
    ('logreg', LogisticRegression(max_iter=600))  # Логистическая регрессия
])

# Параметры для перебора (C - коэффициент регуляризации)
param_grid = {
    'logreg__C': [0.0001, 0.001, 0.01, 0.1, 1]
}

# Настройка GridSearchCV для перебора параметра C
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Обучение модели с перебором параметра C
grid_search.fit(X_train, y_train)

# Лучшая модель и лучший параметр C
best_model = grid_search.best_estimator_
best_C = grid_search.best_params_['logreg__C']

# Предсказания на тестовой выборке
y_test_pred = best_model.predict(X_test)

# Вычисление точности (accuracy) на тестовой выборке
acc_test = accuracy_score(y_test, y_test_pred)

# Вывод самой лучшей метрики на тестовой выборке
print(f"Лучшая метрика на тестовой выборке (accuracy): {acc_test:.2f}")
