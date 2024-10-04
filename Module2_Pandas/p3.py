import pandas as pd
import numpy as np

fruit = np.array(["lemons", "lemons", "lemons", "lemons",
                  "apples", "apples", "apples", "apples",
                  "apples", "apples", "apples"],
                 dtype=object)

shop = np.array(["Shop A", "Shop A", "Shop A", "Shop B",
                 "Shop A", "Shop A", "Shop A", "Shop B",
                 "Shop B", "Shop B", "Shop A"],
                dtype=object)

pl = np.array(["online", "online", "offline",
               "online", "online", "offline",
               "offline", "online", "offline",
               "offline", "offline"],
              dtype=object)

df = pd.DataFrame({'fruit': fruit, 'shop': shop, 'pl': pl,
                   "Q": [1, 2, 2, 3, 3, 4, 5, 6, 7, 4, 4],
                   "P": [5, 4, 5, 5, 6, 6, 8, 9, 9, 3, 3]})
df['total'] = df['Q'] * df['P']
# далее запишите ваш код

pivot = pd.pivot_table(df,
                       values='total',
                       index=['shop'],
                       columns=['pl'],
                       aggfunc='sum')

print(pivot)

print(pivot.iloc[1, 1])
