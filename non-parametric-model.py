import math
import random

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
import tensorflow as tf
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

raw_train_data = pd.read_csv(
    "C:\FCT-NOVA-2024\AA\\assignment1\machine-learning-nova-2024-the-three-body-proble\mlNOVA\mlNOVA\X_train.csv")
raw_train_data['Observation_Id'] = np.repeat(np.arange(5000), 257)

new_array = raw_train_data.to_numpy(dtype=np.float32)
raw_train_data_array = new_array.reshape((5000, 257, 15))


dataframes = []
for i in range(5019):  # 5019
    #if random.random() < 0.1:
    dataframes.append(raw_train_data.loc[i * 256:(i + 1) * 256])
raw_train_data = np.array(dataframes)



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


def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1, 6)):
    best_model = None
    best_error = np.inf
    for k_val in k:
        knn = KNeighborsRegressor(n_neighbors=k_val)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        plot_y_yhat(y_val, y_pred)
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        print(f'K: {k_val} got RMSE of value: {rmse}')
        if rmse < best_error:
            best_model = knn
    return best_model, best_error


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
                # y.append(time_frame[1:13])

    return np.array(x), np.array(y)


x, y = GetXandYLists(raw_train_data)
results = {}
noise = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model, error = validate_knn_regression(x_train, y_train, x_test, y_test)
