import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from plot_y_yhat import plot_y_yhat

def GetXandYLists(data: np.array, account_for_velocity=False):
    x = []
    y = []

    for simulation_matrix in data:
        # print("simulation matrix= ", simulation_matrix)
        init_state_of_simulation = []
        for num, time_frame in enumerate(simulation_matrix):
            # print("time_frame= ", time_frame)
            if num == 0:
                if account_for_velocity:
                    init_state_of_simulation = time_frame[:-2]  # remove the last two elements, because they are ids
                else:
                    init_state_of_simulation = [time_frame[0],  # time
                                                time_frame[1],  # x1
                                                time_frame[2],  # y1
                                                time_frame[5],  # x2
                                                time_frame[6],  # y2
                                                time_frame[9],  # x3
                                                time_frame[10], # y3
                                                ] 
                    
            else:
                init_state_of_simulation[0] = time_frame[0]  # put current time into the vector instead of the 0
                
                x.append(init_state_of_simulation.copy())
                y.append([time_frame[1], time_frame[2], time_frame[5], time_frame[6], time_frame[9], time_frame[10]])

    return np.array(x), np.array(y)


def linearmodel(X_train, y_train, X_test):

    pipe = Pipeline([('std', StandardScaler()),('estimator', LinearRegression())])
    pipe.fit(X_train, y_train)
    plot_y_yhat(y_train, pipe.predict(X_train))
    # print(pipe.predict(X_train).shape)
    rmse = math.sqrt(np.square(np.subtract(y_train,pipe.predict(X_train))).mean())

    # run test
    test_pred = pipe.predict(X_test)
    print(test_pred)

    return rmse, test_pred

data = np.load("X_train.npy", allow_pickle=True)
# print(len(data[0][0]))
x, y = GetXandYLists(data)

print(len(x[0]), len(y[0]))


# prediction form x_test

x_test = pd.read_csv("X_test.csv")
x_test = x_test.drop(['Id'], axis=1)
X_test = np.save("X_test.npy", x_test)
X_test = np.load("X_test.npy", allow_pickle=True)

rms, test_pred = linearmodel(x, y, X_test)
ids = np.arange(1, test_pred.shape[0] + 1).reshape(-1, 1)
y_pred = np.hstack((ids, test_pred))
y_pred_df = pd.DataFrame(y_pred, columns=["id", "x_1", "y_1", "x_2", "y_2", "x_3", "y_3"])
y_pred_df.to_csv("baseline-model.csv", index=False)

print(f"RMSE for linear regression: {rms:.2f}")
# plt.savefig('baseline.pdf')