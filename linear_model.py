import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from scipy import stats
import pprint
import sklearn as sk
from sklearn.metrics import mean_squared_error

data = pd.read_csv(
    "C:\FCT-NOVA-2024\AA\\assignment1\machine-learning-nova-2024-the-three-body-proble\mlNOVA\mlNOVA\X_train.csv")

dataframes = []
print(data.shape)
for i in range(5019):
    dataframes.append(data.loc[i * 256:(i + 1) * 256])
dataframes = np.array(dataframes)


def GetXandYLists(data: np.array):
    x = []
    y = []
    for simulation_matrix in data:
        init_state_of_simulation = []
        for num, time_frame in enumerate(simulation_matrix):
            if num == 0:
                init_state_of_simulation = time_frame
            else:
                init_state_of_simulation[0] = time_frame[0]  # put current time into the vector instead of the 0
                x.append(init_state_of_simulation.copy())
                y.append([time_frame[1], time_frame[2], time_frame[5], time_frame[6], time_frame[9], time_frame[10]])
                #y.append(time_frame[1:13])

    return np.array(x), np.array(y)


def plot_y_yhat(y_val, y_pred, plot_title="plot"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val), MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10, 10))
    for i in range(6):
        x0 = np.min(y_val[idx, i])
        x1 = np.max(y_val[idx, i])
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_val[idx, i], y_pred[idx, i])
        plt.xlabel('True ' + labels[i])
        plt.ylabel('Predicted ' + labels[i])
        plt.plot([x0, x1], [x0, x1], color='red')
        plt.axis('square')
    plt.savefig(plot_title + '.pdf')
    plt.show()


def evaluateGeneralization(X, Y, noise_level):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    plot_y_yhat(y_test, model.predict(X_test))
    return train_mse, test_mse


x, y = GetXandYLists(dataframes)
results = {}
noise = 0.1
train_mse, test_mse = evaluateGeneralization(x, y, noise)
results[noise] = (train_mse, test_mse)
print(f"Noise: {noise} -> Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
