import numpy as np
import random
random.seed(0)
from sklearn.model_selection import train_test_split
from build_polynomial_regression import build_polynomial_regression, evaluate_polynomial_curve
from evaluate_model import evaluate_model
from polynomial import generate_noisy_polynomial_data
from plotting import plot_polynomial_regression, plot_overall_results

# Constants
DEGREE_USED = 20
L1_PENALTY = 0.01 # lambda
L2_PENALTY = 0.01 # lambda
TEST_SIZE = 0.2
DPI = 200

def main():
    X, Y = generate_noisy_polynomial_data()

    curves_data = []

    model_full_no_reg, poly = build_polynomial_regression(X, Y, DEGREE_USED, regularization=None)
    X_plot, Y_plot_poly = evaluate_polynomial_curve(model_full_no_reg, poly, X, Y)
    curves_data.append((X_plot, Y_plot_poly, 'Model trained on the full dataset (for visualization purposes only)'))
    plot_polynomial_regression(X_plot, Y_plot_poly, X, Y, save_path='images/01.png',
                               caption='Polynomial regression without regularization (full dataset)', DPI=DPI)

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

    model_train_no_reg, poly = build_polynomial_regression(X_train, Y_train, DEGREE_USED, regularization=None)
    X_plot, Y_plot_poly = evaluate_polynomial_curve(model_train_no_reg, poly, X, Y)
    # Этот график не добавляем в финальный результат curves_data как неинформативный
    plot_polynomial_regression(X_plot, Y_plot_poly, X_train, Y_train,  X_test, Y_test,
                               save_path='images/02.png',
                               caption='Polynomial regression without regularization (train set only)', DPI=DPI)

    # Test
    square_loss_no_reg = evaluate_model(model_train_no_reg, X_test, Y_test, DEGREE_USED)
    print(f"Square loss on the test set (NO regularization): {square_loss_no_reg}")

    # Train with L1 (Lasso) regularization
    model_train_L1_reg, poly = build_polynomial_regression(X_train, Y_train, DEGREE_USED, 'L1', L1_PENALTY)
    X_plot, Y_plot_poly = evaluate_polynomial_curve(model_train_L1_reg, poly, X, Y)
    curves_data.append((X_plot, Y_plot_poly, 'L1 regularization')) # tuple
    plot_polynomial_regression(X_plot, Y_plot_poly, X_train, Y_train, X_test, Y_test,
                               save_path='images/03.png',
                               caption='Polynomial regression with L1 regularization (Lasso)', DPI=DPI)

    square_loss_L1_reg = evaluate_model(model_train_L1_reg, X_test, Y_test, DEGREE_USED)
    print(f"Square loss on the test set (L1 regularization): {square_loss_L1_reg}")

    # Train with L2 (Ridge) regularization
    model_train_L2_reg, poly = build_polynomial_regression(X_train, Y_train, DEGREE_USED, 'L2', L2_PENALTY)
    X_plot, Y_plot_poly = evaluate_polynomial_curve(model_train_L2_reg, poly, X, Y)
    curves_data.append((X_plot, Y_plot_poly, 'L2 regularization')) # tuple
    plot_polynomial_regression(X_plot, Y_plot_poly, X_train, Y_train, X_test, Y_test,
                               save_path='images/04.png',
                               caption='Polynomial regression with L2 regularization (Ridge)', DPI=DPI)

    square_loss_L2_reg = evaluate_model(model_train_L2_reg, X_test, Y_test, DEGREE_USED)
    print(f"Square loss on the test set (L2 regularization): {square_loss_L2_reg}\n")

    print("Coefficients of the model with no regularization:")
    print(model_full_no_reg.intercept_)
    print(model_full_no_reg.coef_)
    print()
    print("Coefficients of the model with L1 regularization:")
    print(model_train_L1_reg.intercept_)
    print(model_train_L1_reg.coef_) # многие коэффициенты убрались в ноль. Упрощение полинома
    # >>> y = 1.9374-0.8308x^2 - такой полином сделала L2
    print()
    print("Coefficients of the model with L2 regularization:")
    print(model_train_L2_reg.intercept_)
    print(model_train_L2_reg.coef_) # коэффициенты сильно уменьшились, но не обнулились.
    # >>> y = очень_длинное_выражение

    plot_overall_results(curves_data, X_train, Y_train, X_test, Y_test,
                         DPI, save_path='images/05.png')

if __name__ == "__main__":
    main()