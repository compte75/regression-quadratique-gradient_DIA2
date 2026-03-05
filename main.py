import numpy as np
import pandas
import matplotlib.pyplot as plt


df = pandas.read_csv("prix_maisons.csv")

def plot_data(surface, prix):
    plt.figure(figsize=(15, 6))
    plt.scatter(surface, prix, color="red")
    plt.xlabel("Surface (standardisée)")
    plt.ylabel("Prix (standarisée)")
    plt.title("Surface vs Prix")
    plt.show()


def plot(problems, solutions, predictions):
    plt.figure(figsize=(15, 6))
    plt.scatter(problems, solutions, color="red", label="données")
    plt.plot(np.sort(problems), predictions, color="green", label="prédiction")
    plt.legend()
    plt.show()


def quadratic_regression(a, b, c, x):
    return a*x**2 + b*x + c

def mse(y, y_pred):
    return np.mean((y_pred - y)**2)

def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))


def backpropagation_quadratic(a, b, c, x, y, learning_rate):
    n = len(x)
    dL_da = 2/n * np.sum((a * x**2 + b * x + c - y) * x**2)
    dL_db = 2/n * np.sum((a * x**2 + b * x + c - y) * x)
    dL_dc = 2/n * np.sum(a * x**2 + b * x + c - y)
    a -= learning_rate * dL_da
    b -= learning_rate * dL_db
    c -= learning_rate * dL_dc
    predictions = quadratic_regression(a, b, c, x)
    current_rmse = rmse(y, predictions)
    return a, b, c, current_rmse


def gradient_descent_quadratic(x, y, learning_rate=10**(-3), epochs=1000):
    a, b, c = np.random.random(), np.random.random(), np.random.randn()
    rmse_values_per_epoch = []
    for index_epoch in range(epochs):
        a, b, c, current_rmse = backpropagation_quadratic(a, b, c, x, y, learning_rate)
        rmse_values_per_epoch.append(current_rmse)
    x_sorted = np.sort(x)
    predictions = quadratic_regression(a, b, c, x_sorted)
    plot(problems=x_sorted, solutions=y, predictions=predictions)
    return a, b, c, rmse_values_per_epoch


if __name__ == "__main__":
    x_mean, x_std = df["surface"].mean(), df["surface"].std()
    y_mean, y_std = df["prix"].mean(), df["prix"].std()
    df["surface"] = (df["surface"] - x_mean) / x_std
    df["prix"] = (df["prix"] - y_mean) / y_std
    x = df["surface"].values
    y = df["prix"].values
    a, b, c, rmse_history = gradient_descent_quadratic(x, y) 
