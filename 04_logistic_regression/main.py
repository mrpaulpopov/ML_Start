from data import features, labels
from plotting import plot_points, plot_boundary
from scikit_logistic_regression import scikit_logistic_regression

# Plotting the points
plot_points(features, labels)

coefficients_lr, intercept_lr, predictions_lr = scikit_logistic_regression(features, labels)
plot_boundary(features, labels, coefficients_lr, intercept_lr)

print("Logistic Regression Predictions:", predictions_lr)
print("Logistic Regression Coefficients:", coefficients_lr)
print("Logistic Regression Intercept:", intercept_lr)