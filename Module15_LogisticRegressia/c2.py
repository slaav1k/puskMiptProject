import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv("student.csv")

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(columns=["grade"])  # Убираем целевую переменную
y = data["grade"]

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение StandardScaler на тренировочной выборке
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Применение стандартизации к тестовой выборке
X_test_scaled = scaler.transform(X_test)

# Определение min и max для признака age на тестовой выборке
age_min = X_test_scaled[:, X.columns.get_loc("age")].min()
age_max = X_test_scaled[:, X.columns.get_loc("age")].max()

print(f"Минимальное значение age на тестовой выборке: {age_min:.2f}")
print(f"Максимальное значение age на тестовой выборке: {age_max:.2f}")