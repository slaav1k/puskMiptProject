from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=41)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_y_pred)

gb = GradientBoostingClassifier(n_estimators=100,
learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_y_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_y_pred)

print("Accuracy случайного леса:", round(rf_acc, 2))
print("Accuracy градиентного бустинга:", round(gb_acc, 2))