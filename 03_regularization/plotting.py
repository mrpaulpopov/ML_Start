import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0)
from sklearn.preprocessing import PolynomialFeatures


def draw_polynomial(coefs, x):
    """
    Рисование полинома по коэффициентам и массиву х.
    :param coefs:
    :param x:
    :return:
    """
    n = len(coefs)
    y = 0
    for i in range(n):
        y = y + coefs[i] * x ** i # Вычисление полинома по коэффициентам. in range(n) - это для 0, 1, 2.

    plt.plot(x, y, linestyle='-', color='black')
    plt.title(f'Plotting Polynom {coefs}')
    plt.xlim(-1, 1)  # ограничение по оси X
    plt.ylim(1, 2.1)  # ограничение по оси Y
    plt.show()


def plot_polynomial_regression(model, X, Y, degree, X_test=None, Y_test=None, caption=None):
    """
    Рисование полиномиальной регрессии. На входе подается уже готовая модель

    Args:
      model: The trained scikit-learn model object.
      X: Input features (list or numpy array).
      Y: Input labels (list or numpy array).
      degree: The degree of the polynomial used for training.
      X_test: Optional test set features (list or numpy array).
      Y_test: Optional test set labels (list or numpy array).
    """
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    # Plot the original points

    plt.scatter(X, Y, color='blue', label='Original Data')

    # ======================================================================================
    # Generate predicted values for plotting the curve.
    # PolynomialFeatures — это утилита, которая готовит данные для полиномиальной регрессии.
    # Expand x into polynomial features: [x, x^2, ..., x^degree].
    # ======================================================================================

    X_plot = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_plot_poly = poly.fit_transform(X_plot) # Fit on plot data range

    # ================================================================
    # .fit обучает модель, а .predict использует уже обученную модель.
    # Это просто подстановка Х в формулу.
    # ================================================================

    Y_plot_poly = model.predict(X_plot_poly)

    # Plot the polynomial regression curve

    plt.plot(X_plot, Y_plot_poly, color='red', label=f'Polynomial Regression (degree {degree})')

    # Plot test data points if provided
    if X_test is not None and Y_test is not None:
        X_test = np.array(X_test).reshape(-1, 1)
        Y_test = np.array(Y_test)
        plt.scatter(X_test, Y_test, color='orange', marker='^', label='Test Data')

    plt.xlabel('X')
    plt.ylabel('Y')
    if caption is not None:
        plt.title(f'Polynomial Regression (degree {degree})\nmodel={model}, {caption}', fontsize=10)
    else:
        plt.title(f'Polynomial Regression (degree {degree})\nmodel={model}', fontsize=10)
    plt.legend(fontsize="small")
    plt.grid(True)

    # Set plot bounds based on X and Y
    # plt.xlim(np.min(X), np.max(X))
    # plt.ylim(np.min(Y), np.max(Y))

    # Для констистентности внешнего вида между графиками сделаем строгие ограничения:
    plt.xlim(-1, 1)  # ограничение по оси X
    plt.ylim(1, 2.1)  # ограничение по оси Y
    plt.show()