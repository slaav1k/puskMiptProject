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


df2 = df.copy()

subset = df.loc[(df.Q > 3) & (df.shop == 'Shop A')]
# total2 = subset['total'][1]
total2 = subset['total'].iloc[1]
# total2 = df.loc[1, 'total']

fruit_total = df.groupby('fruit')['total'].sum()
fruit_quantity = df.groupby('fruit')['Q'].sum()

# tmp = tmp.loc[(df2.fruit == "lemons")]
lemon_average_price = df.loc[(df2.fruit == "lemons")].groupby('fruit')['P'].mean()
# lemon_average_price = lemon_average_price.loc[(lemon_average_price.fruit == 'lemons')]
print(df)
print(subset)
print(total2)
print(fruit_total)
print(fruit_quantity)
# print(tmp)
print(lemon_average_price)