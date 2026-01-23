from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd # работа с csv
import numpy as np
import matplotlib.pyplot as plt
from linear_regression_dataframe import linear_regression_dataframe

file_path = 'data/Hyderabad.csv'
data = pd.read_csv(file_path)
# print(data.head()) # показ первых 5 строк (по умолчанию). Полезно, чтобы просто быстро взглянуть на таблицу
# print(data.describe()) # быстрый показ статистических данных
# print(data.info()) # понять структуру таблицы

num_rows, num_cols = data.shape
# print("The dataset has ", num_rows, "rows, and ", num_cols, " columns")


def task_linear_regression():
    plt.scatter(data['Area'], data['Price'])
    plt.show()

    # Extract features (X) and target (y)
    X = data[['Area']]  # Feature: Area # 2D, но с одним столбцом. (2434, 1)
    y = data['Price']  # Target: Price  # 1D                       (2434,)

    model, area_range, predicted_prices = linear_regression_dataframe(X, y)

    print(f"y-intercept: {model.intercept_}")
    print(f"slope (coefficient of Area): {model.coef_[0]}")

    # Plot the original data curves_data
    plt.scatter(X, y, color='blue', label='Data Points')

    # Plot the regression line
    plt.plot(area_range, predicted_prices, color='red', linewidth=2, label='Regression Line')

    # Add labels and title
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.title('Area vs. Price with Linear Regression')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    # Код выдаст предупреждение X does not have valid feature names, но это не страшно.
    # y - именованные значения в pandas: y = data['Price'],
    # x - это сгенерированный массив linspace из numpy.
    # Предупреждение ругается на то, что у х нет имен, и это может привести к путанице.

def task_save_for_latter():
    # https://github.com/luisguiserrano/manning/blob/master/Chapter_03_Linear_Regression/House_price_predictions.ipynb
    # print(data)

    # СМОТРИМ таблицу вручную, оцениваем валидность данных.
    # Видим, что в последних строках вместо единиц и нулей "9".
    # Pandas об этом не знает, знаем только мы, поэтому срезаем вручную.
    # [:2434] берет только первые 2434 строки.
    data_truncated = data[:2434]
    # print(data_truncated)

    # .copy - это полная копия данных.
    # Если без .copy, то это получится ссылка, нам это не нужно.
    data_scaled = data_truncated.copy()

    # z-score for Area
    area_mean = data_scaled['Area'].mean()
    area_std = data_scaled['Area'].std()
    data_scaled['Area'] = (data_scaled['Area'] - area_mean) / area_std

    # z-score for No. of Bedrooms
    bedrooms_mean = data_scaled['No. of Bedrooms'].mean()
    bedrooms_std = data_scaled['No. of Bedrooms'].std()
    data_scaled['No. of Bedrooms'] = (data_scaled['No. of Bedrooms'] - bedrooms_mean) / bedrooms_std

    # Print the head of the new dataframe to verify
    # print(data_scaled.head())
    # print(data_scaled['No. of Bedrooms'])

    # One-hot encoding (pd.get_dummies):
    # Это когда текстовый столбец (признак) кодируется числовыми значениями.
    # Кодировать как 1, 2, 3,.. неправильно, поэтому столбец дублируется в несколько столбцов на:
    # Location_Downtown, Location_Suburb итд.
    # и они кодируется 1 и 0.

    data_scaled_encoded = pd.get_dummies(data_scaled, columns=['Location'], prefix='Location', dtype=int)
    # print(data_scaled_encoded.head())



    # Separate features (X) and target (y) from the scaled and encoded data
    # 'Price' is the target variable
    X_full = data_scaled_encoded.drop('Price', axis=1)
    y_full = data_scaled_encoded['Price']

    # Create a Linear Regression model
    model_predict_all = LinearRegression()

    # Fit the model to the data
    model_predict_all.fit(X_full, y_full)

    # Print the coefficients of the model

    print("\nLinear Regression Model Coefficients (Predicting Price from all features):")
    print(f"Intercept: {model_predict_all.intercept_}")
    print("Coefficients for features:")
    for feature, coef in zip(X_full.columns, model_predict_all.coef_):
        print(f"{feature}: {coef}")

task_linear_regression()
