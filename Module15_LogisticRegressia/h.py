import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Загрузка данных
data = pd.read_csv("taxi.csv")

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(columns=["tip_class"])  # Убираем целевую переменную
y = data["tip_class"]  # Целевая переменная

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Инициализация логистической регрессии с max_iter=400
model = LogisticRegression(max_iter=400)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели
model.fit(X_train_scaled, y_train)

# Предсказания на тестовой выборке
y_test_pred = model.predict(X_test_scaled)

# Рассчитываем accuracy для тренировочной и тестовой выборок
acc_train = accuracy_score(y_train, model.predict(X_train_scaled))
acc_test = accuracy_score(y_test, y_test_pred)

# Вывод метрик accuracy
print(f"accuracy на тренировочной выборке: {acc_train:.2f}")
print(f"accuracy на тестовой выборке: {acc_test:.2f}")

# Получаем матрицу ошибок
cm = confusion_matrix(y_test, y_test_pred, labels=data["tip_class"].unique())

# Переводим матрицу ошибок в DataFrame для удобства
cm_df = pd.DataFrame(cm, columns=data["tip_class"].unique(), index=data["tip_class"].unique())

# Рассчитываем процент ошибочных предсказаний для каждого класса
errors = cm_df.apply(lambda x: (x.sum() - x[x.name]) / x.sum() * 100, axis=1)

# Выводим ошибочные предсказания по каждому классу
for label, error_percentage in errors.items():
    print(f"{label}: {error_percentage:.2f}% ошибочных предсказаний")
