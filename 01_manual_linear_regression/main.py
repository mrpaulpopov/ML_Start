import matplotlib.pyplot as plt
from data_simple import x, y
from plotting import plot_points, draw_line
from manual_linear_regression import manual_linear_regression
from scikit_linear_regression import scikit_linear_regression


plt.ylim(0,500) # ограничение внешнего вида графика

if __name__ == "__main__":
    m, b, errors = manual_linear_regression(x, y, learning_rate=0.01,
                                            epochs=2000)  # здесь в т.ч. идет постройка графика
    print('Price per room:', m)
    print('Base price:', b)
    plot_points(x, y)
    plt.show() # точки + итерационные прямые из linear regression

    plot_points(x, y) # график исходных данных
    draw_line(m, b, starting=0, ending=8)
    plt.show() # точки + итоговая прямая

    plt.plot(range(len(errors)), errors) # график ошибок. По нему делаем вывод, что epoch = 2000 достаточно
    plt.show() # график ошибок

    # =================================
    # range(len(errors)) - это явное указание оси X.
    # Если сделать plt.plot(errors), он возьмет Х из Y.
    # len(errors) = 3
    # range(len(errors)) делает список до 3: [0, 1, 2]
    # (То есть вручную считает кол-во элементов в errors и нумерует их в range)
    # Это будет удобно, когда мы захотим изменить Х, взять их не из errors.
    # =================================

    model, y_hat, y_hat_array = scikit_linear_regression(x, y, point=4)

    # Print the coefficients and intercept
    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)

    plt.scatter(x, y, color='blue', label='Original Data')
    plt.plot(x, y_hat_array, color='red', label='Regression Line')
    plt.xlabel('Features')
    plt.ylabel('Labels')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
