import csv
import pandas as pd

data = pd.read_csv("police.csv", sep=",")


countsNull = data.isnull().sum()


max_null_column = countsNull.idxmax()
# max_null_count = countsNull.max()

data.drop(max_null_column, inplace=True, axis=1)
print(data.driver_gender.str.contains("M").count())
