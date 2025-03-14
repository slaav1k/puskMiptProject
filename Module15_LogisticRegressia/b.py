import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Загрузка датасета
df = pd.read_csv('student.csv')

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop(columns=['grade'])  # Признаки
y = df['grade']  # Целевая переменная

# Разбиение на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Инициализация модели логистической регрессии
model = LogisticRegression(max_iter=1000)

# Обучение модели
model.fit(X_train, y_train)

# Получение веса для признака studytime
studytime_weight = model.coef_[0][X.columns.get_loc('studytime')]

# Округление до 2 знаков
studytime_weight_rounded = round(studytime_weight, 2)

# Вывод результата
print(studytime_weight_rounded)
print(sklearn.__version__)
