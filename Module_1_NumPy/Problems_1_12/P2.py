import numpy as np

some_data = [
    [3, 8, 1, 0, 1, 2],
    [9, 2, 7, 3, 0, 4],
    [2, 5, 1, 3, 1, 8],
    [5, 1, 2, 1, 1, 0]
]

lines = np.array(some_data)[1:]
last_column = np.array(some_data)[:, -1]
array_t = np.array(some_data).transpose()
array_sum = np.sum(np.array(some_data))
array_avg = np.average(np.array(some_data))

print(array_avg)
