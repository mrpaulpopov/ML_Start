import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors) # curves colors


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


def plot_polynomial_regression(X_plot, Y_plot_poly, X_train, Y_train, X_test=None, Y_test=None,
                               save_path=None, caption=None, DPI=200):
    """
    Рисование полиномиальной регрессии. На входе подается уже готовая модель

    Args:
      X: Input features (numpy array).
      Y: Input labels (numpy array).
      degree: The degree of the polynomial used for training.
      X_test: Optional test set features (list or numpy array).
      Y_test: Optional test set labels (list or numpy array).
    """
    plt.figure(dpi=DPI)

    # Plot the original curves_data
    plt.scatter(X_train, Y_train, color='black', label='Training Data')

    # Plot the polynomial regression curve
    plt.plot(X_plot, Y_plot_poly, color='indianred', label=f'Polynomial Regression')

    # Plot test data curves_data if provided
    if X_test is not None and Y_test is not None:
        X_test = np.array(X_test).reshape(-1, 1)
        Y_test = np.array(Y_test)
        plt.scatter(X_test, Y_test, color='chocolate', marker='^', label='Test Data')

    plt.xlabel('X')
    plt.ylabel('Y')
    if caption is not None:
        plt.title(caption, fontsize=10)
    else:
        plt.title(f'Polynomial regression', fontsize=10)
    plt.legend(fontsize="small")
    plt.grid(True)

    # Set plot bounds based on X and Y
    # plt.xlim(np.min(X), np.max(X))
    # plt.ylim(np.min(Y), np.max(Y))

    # Для констистентности внешнего вида между графиками сделаем строгие ограничения:
    plt.xlim(-1, 1)  # ограничение по оси X
    plt.ylim(1, 2.1)  # ограничение по оси Y

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_overall_results(curves_data, X_train, Y_train, X_test, Y_test, DPI, save_path):
    """
     Рисование полиномиальной регрессии. На входе подается уже готовая модель

     Args:
       model: The trained scikit-learn model object.
       X: Input features (list or numpy array).
       Y: Input labels (list or numpy array).
       X_test: Optional test set features (list or numpy array).
       Y_test: Optional test set labels (list or numpy array).
     """
    plt.figure(dpi=DPI)
    # Plot the original curves_data
    plt.scatter(X_train, Y_train, color='black', label='Training Data')
    # Plot test data curves_data if provided
    if X_test is not None and Y_test is not None:
        X_test = np.array(X_test).reshape(-1, 1)
        Y_test = np.array(Y_test)
        plt.scatter(X_test, Y_test, color='chocolate', marker='^', label='Test Data')

    # Plot the polynomial regression curve
    for i, curve in enumerate(curves_data):
        if i == 0:
            plt.plot(curve[0], curve[1], color='black', label=curve[2],
                     linestyle=':')
        else:
            plt.plot(curve[0], curve[1], label=curve[2])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Comparison of polynomial regression models', fontsize=10)
    plt.legend(fontsize="small")
    plt.grid(True)

    # Set plot bounds based on X and Y
    # plt.xlim(np.min(X), np.max(X))
    # plt.ylim(np.min(Y), np.max(Y))

    # Для констистентности внешнего вида между графиками сделаем строгие ограничения:
    plt.xlim(-1, 1)  # ограничение по оси X
    plt.ylim(1, 2.1)  # ограничение по оси Y

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
