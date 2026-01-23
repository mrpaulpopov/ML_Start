import numpy as np
import random
random.seed(0)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge


def build_polynomial_regression(X, Y, degree, regularization=None, alpha=1.0):
    """
    Trains a polynomial regression model with optional L1 or L2 regularization.

    Args:
      X: Input features (list or numpy array).
      Y: Input labels (list or numpy array).
      degree: The degree of the polynomial.
      regularization: Type of regularization ('L1', 'L2', or None). Defaults to None.
      alpha: Regularization strength (for L1 and L2). Defaults to 1.0.

    Returns:
      model: A trained scikit-learn model object,
    """

    # настройки model
    if regularization == 'L1':
        model = Lasso(alpha=alpha)
    elif regularization == 'L2':
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()

    # ======================================================================================
    # PolynomialFeatures — это утилита, которая готовит данные для полиномиальной регрессии:
    # expand x into polynomial features: [x, x^2, ..., x^degree].
    # ======================================================================================

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model.fit(X_poly, Y)  # применение модели к X_poly и Y

    return model, poly

def evaluate_polynomial_curve(model, poly, X, Y):
    '''
    X, Y - это 40 исходных точек.
    X_plot - это 100 красивых точек, просто для графика.
    .fit обучает модель, а .predict использует уже обученную модель
    (это просто подстановка Х в модель).
    X_plot - обычные числа, идут на график.
    X_plot_poly - промежуточная версия, X_plot после применения модели poly.
    Y_plot_poly - после обработки X_plot_poly это обычные числа Y, идут на график.
    '''

    X_plot = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)  # Преобразуем X_plot в вид для модели. Это служебная версия для модели.
    Y_plot_poly = model.predict(X_plot_poly)  # просто подставляем X в модель и получаем Y.
    return X_plot, Y_plot_poly
