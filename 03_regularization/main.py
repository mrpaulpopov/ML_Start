import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0)
from sklearn.model_selection import train_test_split
from plotting import plot_polynomial_regression
from train_polynomial_regression import train_polynomial_regression
from evaluate_model import evaluate_model
from polynomial import polynomial
from plotting import draw_polynomial

# Constants
DEGREE_USED = 20
L1_PENALTY = 0.01 # lambda
L2_PENALTY = 0.01 # lambda
TEST_SIZE = 0.2

# Отделяем математику от эксперимента, модели и визуализации!

# Исходный полином -x^2+2

coefs = [2,0,-1] # это полином, записанный в виде коэффициентов a0, a1, ...  Это короткий способ записать любой полином.
x = np.linspace(-1, 1, 1000) # от -1 до 1 создаем 1000 точек.
draw_polynomial(coefs, x) # построим параболу

# Сделаем 40 точек, рандомно отодвинем от графика

X = []
Y = []
for i in range(40): # создаем 40 точек для X и Y
    x = random.uniform(-1,1) # X - это случайное распределение между -1 и 1 (40 раз, 40 точек)
    y = polynomial(coefs, x) + random.gauss(0,0.1)
    X.append(x)
    Y.append(y)

model_full_no_reg = train_polynomial_regression(X, Y, DEGREE_USED, regularization=None)
# Теперь мы имеем модель. Дальше сделаем .predict и нарисуем:
plot_polynomial_regression(model_full_no_reg, X, Y, DEGREE_USED, caption="Full dataset (1)")

# ============================================================================
# Разделение на тестовый набор и набор для обучения:
# На входе X, Y - features, labels.
# train_test_split отделяет случайным образом данные
# на тестовый набор и набор для обучения.
# test_size = 0.2 выбирает для обучения случайные 20% данных, т.е. пар (X, Y).
# random_state - это seed
# На выходе пары X_train, Y_train - для обучения, X_test, Y_test - для теста.
# ============================================================================

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=0)

print(f'\nShape of X_train: {np.shape(X_train)}, Shape of Y_train: {np.shape(Y_train)}')
print(f'Shape of X_test: {np.shape(X_test)}, Shape of Y_test: {np.shape(Y_test)}\n')

model_train_no_reg = train_polynomial_regression(X_train, Y_train, DEGREE_USED, regularization=None)
# Теперь мы имеем модель. Дальше сделаем .predict и нарисуем:
plot_polynomial_regression(model_train_no_reg, X_train, Y_train, DEGREE_USED,  X_test, Y_test, caption=f'Training dataset ({1-TEST_SIZE})')

# Test
square_loss_no_reg = evaluate_model(model_train_no_reg, X_test, Y_test, DEGREE_USED)
print(f"Square loss on the test set (NO regularization): {square_loss_no_reg}")

# Train with L1 (Lasso) regularization
model_L1_reg = train_polynomial_regression(X_train, Y_train, DEGREE_USED, 'L1', L1_PENALTY)
plot_polynomial_regression(model_L1_reg, X_train, Y_train, DEGREE_USED, X_test, Y_test)

square_loss_L1_reg = evaluate_model(model_L1_reg, X_test, Y_test, DEGREE_USED)
print(f"Square loss on the test set (L1 regularization): {square_loss_L1_reg}")

# Train with L2 (Ridge) regularization
model_L2_reg = train_polynomial_regression(X_train, Y_train, DEGREE_USED, 'L2', L2_PENALTY)
plot_polynomial_regression(model_L2_reg, X_train, Y_train, DEGREE_USED, X_test, Y_test)

square_loss_L2_reg = evaluate_model(model_L2_reg, X_test, Y_test, DEGREE_USED)
print(f"Square loss on the test set (L2 regularization): {square_loss_L2_reg}\n")

print("Coefficients of the model with no regularization:")
print(model_full_no_reg.intercept_)
print(model_full_no_reg.coef_)
print()
print("Coefficients of the model with L1 regularization:")
print(model_L1_reg.intercept_)
print(model_L1_reg.coef_) # многие коэффициенты убрались в ноль. Упрощение полинома
# >>> y = 1.9374-0.8308x^2 - такой полином сделала L2
print()
print("Coefficients of the model with L2 regularization:")
print(model_L2_reg.intercept_)
print(model_L2_reg.coef_) # коэффициенты сильно уменьшились, но не обнулились.
# >>> y = очень_длинное_выражение

