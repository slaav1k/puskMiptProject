import numpy as np
import pandas as pd

df = pd.DataFrame(np.nan, index=[0, 1, 2, 3], columns=['I', 'II', 'III'])
df.loc[0, "I"] = 1
df.loc[1, "I"] = 2
df.loc[2, "I"] = 3
df.loc[3, "I"] = 4
df.loc[0, "II"] = 5
df.loc[1, "II"] = 6
df.loc[2, "III"] = 7
df.loc[3, "III"] = 6


df = df.set_axis([1, 2, 3, 4], axis="index")
df = df.set_axis(["A", "B", "C"], axis="columns")
df.fillna(55, inplace=True)

print(df)