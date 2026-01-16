import numpy as np
import random
random.seed(0)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge

def train_polynomial_regression(X, Y, degree, regularization=None, alpha=1.0):
    """
    Trains a polynomial regression model with optional L1 or L2 regularization.

    Args:
      X: Input features (list or numpy array).
      Y: Input labels (list or numpy array).
      degree: The degree of the polynomial.
      regularization: Type of regularization ('L1', 'L2', or None). Defaults to None.
      alpha: Regularization strength (for L1 and L2). Defaults to 1.0.

    Returns:
      На выходе обученная модель, это объект scikit-learn.
      A trained scikit-learn model object (LinearRegression, Ridge, or Lasso).
    """

    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    # ======================================================================================
    # PolynomialFeatures — это утилита, которая готовит данные для полиномиальной регрессии.
    # Expand x into polynomial features: [x, x^2, ..., x^degree].
    # ======================================================================================

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    if regularization == 'L1':
        model = Lasso(alpha=alpha) # настройки model
    elif regularization == 'L2':
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()

    model.fit(X_poly, Y) # применение модели к X_poly и Y

    return model