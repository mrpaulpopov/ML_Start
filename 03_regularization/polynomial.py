import numpy as np
import random
random.seed(0)

def polynomial(coefs, x):
    n = len(coefs)
    y = 0
    for i in range(n):
        y = y + coefs[i] * x ** i  # Вычисление полинома по коэффициентам. in range(n) - это для 0, 1, 2.
    return y

def generate_noisy_polynomial_data():
    # Исходный полином -x^2+2
    coefs = [2, 0,
             -1]  # это полином, записанный в виде коэффициентов a0, a1, ...  Это короткий способ записать любой полином.
    x = np.linspace(-1, 1, 1000)  # от -1 до 1 создаем 1000 точек.
    # If we want to draw the original function:
    # draw_polynomial(coefs, x)

    # Сделаем 40 точек, рандомно отодвинем от графика
    X = []
    Y = []
    for i in range(40):  # создаем 40 точек для X и Y
        x = random.uniform(-1, 1)  # X - это случайное распределение между -1 и 1 (40 раз, 40 точек)
        y = polynomial(coefs, x) + random.gauss(0, 0.1)
        X.append(x)
        Y.append(y)

    # поворачиваем X в вертикальный вид
    X = np.array(X).reshape(-1, 1)  # (40,) -> (40,1)
    Y = np.array(Y)  # Y был list, а стал numpy-объект
    return X, Y