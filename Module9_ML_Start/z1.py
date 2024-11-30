import numpy as np
from sklearn.linear_model import LinearRegression

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([6, 7, 8, 9, 10])
y = np.array([11, 12, 13, 14, 15])
reg = LinearRegression()

arr = np.column_stack((x1, x2))
reg.fit(arr, y)
reg.coef_ = np.array([round(reg.coef_[0], 16), round(reg.coef_[1], 16)])

# w = reg.coef_
# print(w)
print(reg.coef_[0] == 0.5)
print(reg.coef_[1] == 0.5000000000000001)
print(reg.coef_[0])
print(reg.coef_[1])
