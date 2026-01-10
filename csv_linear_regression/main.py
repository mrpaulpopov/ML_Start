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
print("The dataset has ", num_rows, "rows, and ", num_cols, " columns")

plt.scatter(data['Area'], data['Price'])
plt.show()

# Extract features (X) and target (y)
X = data[['Area']]  # Feature: Area
y = data['Price']  # Target: Price

model, area_range, predicted_prices = linear_regression_dataframe(X, y)

print(f"y-intercept: {model.intercept_}")
print(f"slope (coefficient of Area): {model.coef_[0]}")

# Plot the original data points
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