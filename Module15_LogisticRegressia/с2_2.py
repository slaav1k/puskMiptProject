import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sklearn  # Для версии

# Загрузка датасета
df = pd.read_csv('student.csv')

# Проверка наличия столбца 'grade'
if 'grade' not in df.columns:
    raise ValueError("Столбец 'grade' отсутствует в данных.")

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop(columns=['grade'])  # Признаки
y = df['grade']  # Целевая переменная

# Проверка наличия столбца 'age'
if 'age' not in X.columns:
    raise ValueError("Столбец 'age' отсутствует в данных.")

# Разбиение на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# Масштабирование данных с помощью StandardScaler
scaler = StandardScaler()

# Обучение стандартизатора на тренировочной выборке и преобразование данных
X_train_scaled = scaler.fit_transform(X_train)

# Преобразование тестовой выборки с использованием тех же параметров стандартизации
X_test_scaled = scaler.transform(X_test)

# Проверка на NaN или бесконечные значения
if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
    raise ValueError("Масштабированные данные содержат NaN или бесконечные значения.")

# Проверка минимального и максимального значения для признака 'age' на тестовой выборке
age_index = list(X.columns).index('age')
age_min = X_test_scaled[:, age_index].min()
age_max = X_test_scaled[:, age_index].max()

# Вывод результата
print(f"Min age on test set after scaling: {age_min}")
print(f"Max age on test set after scaling: {age_max}")

# Вывод версии scikit-learn
print("scikit-learn version:", sklearn.__version__)