import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn  # Для версии

# Загрузка датасета
df = pd.read_csv('student.csv')

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop(columns=['grade'])  # Признаки
y = df['grade']  # Целевая переменная

# Разбиение на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Масштабирование данных с помощью StandardScaler
scaler = StandardScaler()

# Обучение стандартизатора на тренировочной выборке и преобразование данных
X_train_scaled = scaler.fit_transform(X_train)

# Преобразование тестовой выборки с использованием тех же параметров стандартизации
X_test_scaled = scaler.transform(X_test)

# Проверка минимального и максимального значения для признака 'age' на тестовой выборке
age_min = X_test_scaled[:, X.columns.get_loc('age')].min()
age_max = X_test_scaled[:, X.columns.get_loc('age')].max()

# Вывод результата
print(f"Min age on test set after scaling: {age_min}")
print(f"Max age on test set after scaling: {age_max}")

# Вывод версии scikit-learn
print("scikit-learn version:", sklearn.__version__)
