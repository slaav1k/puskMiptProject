import numpy as np
from sklearn.linear_model import LinearRegression
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([6, 7, 8, 9, 10])
y = np.array([11, 12, 13, 14, 15])
reg = LinearRegression()

arr = np.column_stack((x1, x2))
reg.fit(arr, y)
ans = reg.predict([[5, 6]])[0]
print(ans)