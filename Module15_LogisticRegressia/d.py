import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv("student.csv")

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(columns=["grade"])  # Убираем целевую переменную
y = data["grade"]

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных с помощью StandardScaler
scaler = StandardScaler()

# Обучение стандартизатора на тренировочной выборке и преобразование данных
X_train_scaled = scaler.fit_transform(X_train)

# Преобразование тестовой выборки с использованием тех же параметров стандартизации
X_test_scaled = scaler.transform(X_test)

# Обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Предсказания на тренировочной и тестовой выборках
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Вычисление точности (accuracy) на тренировочной и тестовой выборках
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# Вывод результатов, округленных до двух знаков
print(f"Accuracy на тренировочной выборке: {acc_train:.2f}")
print(f"Accuracy на тестовой выборке: {acc_test:.2f}")
print(f"Accuracy на тренировочной выборке: {acc_train}")
print(f"Accuracy на тестовой выборке: {acc_test}")
