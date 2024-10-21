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

data = np.load("data.npy", allow_pickle=True)
np.random.shuffle(data)

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
    plt.savefig("knn-plots/k_" + plot_title + '.pdf')
    plt.show()


def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1, 15)):
    best_model = None
    best_error = np.inf
    for k_val in k:
        knn = KNeighborsRegressor(n_neighbors=k_val)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        plot_y_yhat(y_val, y_pred, str(k_val))
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        print(f'K: {k_val} got RMSE of value: {rmse}')
        if rmse < best_error:
            best_model = knn
    return best_model, best_error


def GetXandYLists(data: np.array, account_for_velocity=False):
    x = []
    y = []
    for simulation_matrix in data:
        init_state_of_simulation = []
        for num, time_frame in enumerate(simulation_matrix):
            if num == 0:
                if account_for_velocity:
                    init_state_of_simulation = time_frame[:-2]  # remove the last two elements, because they are ids
                else:
                    init_state_of_simulation = [time_frame[0],  # time
                                                time_frame[1],  # x1
                                                time_frame[2],
                                                time_frame[3],  # x2
                                                time_frame[4],
                                                time_frame[5],  # x3
                                                time_frame[6]]
            else:
                init_state_of_simulation[0] = time_frame[0]  # put current time into the vector instead of the 0
                x.append(init_state_of_simulation.copy())
                y.append([time_frame[1], time_frame[2], time_frame[3], time_frame[4], time_frame[5], time_frame[6]])

    return np.array(x), np.array(y)



x, y = GetXandYLists(data)
print(x.shape, y.shape)
results = {}
noise = 0.1
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)


model, error = validate_knn_regression(x_train, y_train, x_val, y_val)
raw_test_data = pd.read_csv("X_test.csv")
raw_test_data.drop(columns=['Id'], inplace=True)

raw_test_data = np.array(raw_test_data)
X_test = np.save("X_test.npy", x_test)
X_test = np.load("X_test.npy", allow_pickle=True)
test_pred = model.predict(raw_test_data)
ids = np.arange(0, test_pred.shape[0]).reshape(-1, 1)
y_pred = np.hstack((ids, test_pred))
y_pred_df = pd.DataFrame(y_pred, columns=["Id", "x_1", "y_1", "x_2", "y_2", "x_3", "y_3"])
y_pred_df['Id'] = y_pred_df['Id'].astype(np.int32)
y_pred_df.to_csv("knn_submission.csv", index=False)

