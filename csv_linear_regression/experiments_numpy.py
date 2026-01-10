import numpy as np

# # 1D массив
# x = np.array([1, 2, 3, 4, 5])
# print("x (1D):", x)
# print("shape:", x.shape)
#
# # Превращаем в 2D массив (5 строк, 1 столбец)
# x_2d = x.reshape(-1, 1)
# print("\nx_2d (2D):\n", x_2d)
# print("shape:", x_2d.shape)
#
# # Превращаем в 2D массив (1 строка, 5 столбцов)
# x_2d_row = x.reshape(1, -1)
# print("\nx_2d_row (1x5):\n", x_2d_row)
# print("shape:", x_2d_row.shape)


# # linspace - генерация равномерно распределенных чисел
# arr = np.linspace(1, 10, 11)
# print("linspace 1D:", arr)
# print("shape:", arr.shape)
#
# # Превращаем в столбец (10,1)
# arr_col = arr.reshape(-1, 1)
# print("\nlinspace column:\n", arr_col)
# print("shape:", arr_col.shape)


# Два признака: площадь и количество комнат
area = np.linspace(50, 300, 5)  # 5 значений площади
rooms = np.linspace(1, 5, 5)    # 5 значений комнат

print("area:", area)
print("rooms:", rooms)

# Объединяем в 2D массив (5 образцов, 2 признака)
X = np.column_stack((area, rooms))
print("\nX (5x2):\n", X)
print("shape:", X.shape)
