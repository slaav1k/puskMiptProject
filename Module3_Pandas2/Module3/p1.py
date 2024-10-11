# начальный код

import random
import pandas as pd

random.seed(10)

list_metrics = []

for i in range(0, 300):
    n = random.randint(-100, 1000)
    list_metrics.append(n)

list_metrics = pd.DataFrame(list_metrics)
# print(list_metrics)

result1 = float(round(list_metrics.std().iloc[0], 2))
result2 = float(round((list_metrics.max().iloc[0] - list_metrics.min()).iloc[0], 2))
print(result1)
print(result2)
