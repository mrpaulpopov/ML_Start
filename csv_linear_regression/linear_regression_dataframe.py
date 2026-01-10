from sklearn.linear_model import LinearRegression
import numpy as np

def linear_regression_dataframe(X, y, point=4):
    model = LinearRegression()
    model.fit(X, y) # Fit the model to the data

    # Predict prices for a range of Area values to plot the regression line
    # X.min и X.max - это мин. и макс. значения столбца Area.
    # np.linspace создает 100 ровно распределенных чисел между X.min и X.max.
    # .reshape(-1, 1) превращает это в 1 столбец, а кол-во строк адаптирует
    area_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    predicted_prices = model.predict(area_range)

    return model, area_range, predicted_prices