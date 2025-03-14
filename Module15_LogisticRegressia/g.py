import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv("taxi.csv")

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(columns=["tip_class"])  # Убираем целевую переменную
y = data["tip_class"]  # Целевая переменная

# Разделение на тренировочную и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Масштабирование данных, если необходимо
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели логистической регрессии с параметром max_iter=400
model = LogisticRegression(max_iter=400)
model.fit(X_train_scaled, y_train)

# Предсказания на тренировочной и тестовой выборках
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Вычисление метрики accuracy на тренировочной и тестовой выборках
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# Вывод метрик
print(f"Accuracy на тренировочной выборке: {acc_train:.2f}")
print(f"Accuracy на тестовой выборке: {acc_test:.2f}")
