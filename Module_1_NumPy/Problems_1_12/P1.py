import numpy as np

array = np.arange(10) ** 4
array_2 = np.arange(10) ** 3

print(array)
print(array_2)

array_sum = np.sum(array) + np.sum(array_2)
array_difference = np.sum(array) - np.sum(array_2)
game_payments2 = array[1]
subscription_last = array_2[-1]

print(array_sum)
print(np.sum(array))
