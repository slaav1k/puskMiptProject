import numpy as np

# создаем исходную матрицу A размером 5x5
A = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25]])

# создаем матрицу фильтра F размером 3x3
F = np.array([[1, 2, -1],
              [1, 0, -1],
              [1, 0, -1]])

# определяем размеры матриц A и F
height_a, width_a = A.shape
height_f, width_f = F.shape

# инициализируем выходную матрицу C
output_height = height_a - height_f + 1
output_width = width_a - width_f + 1
C = np.zeros((output_height, output_width))

# проходим по каждой строке матрицы A
for i in range(output_height):
    # проходим по каждому столбцу матрицы A
    for j in range(output_width):
        # получаем окно размером 3x3, начиная с текущей позиции
        window = A[i:i + height_f, j:j + width_f]
        # перемножаем окно с матрицей фильтра
        result = window * F
        # суммируем результаты умножения, чтобы получить новое значение в выходной матрице C
        C[i, j] = np.sum(result)

print(C)
