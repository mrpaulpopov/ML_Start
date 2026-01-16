def polynomial(coefs, x):
    n = len(coefs)
    y = 0
    for i in range(n):
        y = y + coefs[i] * x ** i  # Вычисление полинома по коэффициентам. in range(n) - это для 0, 1, 2.
    return y