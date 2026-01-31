from sklearn.linear_model import LogisticRegression

def scikit_logistic_regression(features, labels):
    logistic_regression = LogisticRegression() # Create a Logistic Regression model
    logistic_regression.fit(features, labels) # Fit the model to the data

    # Checks that the model works: predict the labels for the features
    predictions_lr = logistic_regression.predict(features)

    # Get coefficients and intercept from the fitted model
    coefficients_lr = logistic_regression.coef_[0]
    intercept_lr = logistic_regression.intercept_[0]

    return coefficients_lr, intercept_lr, predictions_lr