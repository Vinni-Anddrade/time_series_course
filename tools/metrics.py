import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error


def measuring_predictions(y_true: pd.Series, y_pred: pd.Series):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = 1 - mape

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
    print(f"Model's Accuracy: {accuracy * 100:.2f}%")


def model_assessment(train, test, predictions, chart_title: None):
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Training Data")
    plt.plot(test, label="Testing Data")
    plt.plot(predictions, label="Prediction Data", ls="--")
    plt.title(chart_title, loc="left")
    plt.xlabel("Periods")
    plt.ylabel("Complaints")
    plt.legend()
    plt.tight_layout()
    plt.show()
