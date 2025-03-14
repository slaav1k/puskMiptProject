import pandas as pd
from sklearn.model_selection import train_test_split
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
    ('logreg', LogisticRegression(penalty='none', max_iter=1000))  # Логистическая регрессия без регуляризации
])

# Обучение пайплайна на тренировочной выборке
pipeline.fit(X_train, y_train)

# Предсказания на тренировочной и тестовой выборках
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Вычисление точности (accuracy) на тренировочной и тестовой выборках
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# Вывод результатов, округленных до двух знаков
print(f"Accuracy на тренировочной выборке: {acc_train:.2f}")
print(f"Accuracy на тестовой выборке: {acc_test:.2f}")
print(f"Accuracy на тренировочной выборке: {acc_train}")
print(f"Accuracy на тестовой выборке: {acc_test}")