import numpy as np

# Загрузка файла
data = np.load('images.npy')

# Вывод информации о массиве (форма, тип данных)
print(data.shape, data.dtype)

# Вывод самого содержимого массива
print(data)
