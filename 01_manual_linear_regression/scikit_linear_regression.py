from sklearn.linear_model import LinearRegression
import numpy as np

def scikit_linear_regression(x, y, point=4):
    # Reshape the features to be a 2D array, which is required by scikit-learn
    x_reshaped = x.reshape(-1, 1) # это поворот из горизонтального 1D-массива в вертикальный 1D-массив, требование scikit-learn

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(x_reshaped, y)

    # Make a prediction for a new point
    new_point = np.array([[point]])  # Predict for a feature value of 4
    y_pred = model.predict(new_point)
    print(f"Predicted label for x={point}: {y_pred}")

    # Generate the predicted values using the model
    y_pred_array = model.predict(x_reshaped)
    print(y_pred_array)

    return model, y_pred, y_pred_array

# ===========================
# Scikit требователен к входным данным, поэтому превращаем в массивы через [[ ]]
# ===========================