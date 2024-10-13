import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    "C:\FCT-NOVA-2024\AA\\assignment1\machine-learning-nova-2024-the-three-body-proble\mlNOVA\mlNOVA\X_train.csv")

dataframes = []
for i in range(5019):  # 5019
    if random.random() < 0.01:
        dataframes.append(data.loc[i * 256:(i + 1) * 256])
dataframes = np.array(dataframes)


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


def validate_poly_regression(X_train, y_train, X_val, y_val,
                             regressor=None, degrees=range(1, 4),
                             max_features=None):
    best_model = None
    best_error = np.inf
    if regressor is None:
        alphas = np.logspace(-6, 6, 13)
        regressor = RidgeCV(alphas=alphas)
    for deg in degrees:
        polyreg = make_pipeline(PolynomialFeatures(deg),
                                StandardScaler(),
                                regressor)
        #polyreg = make_pipeline([('Polynomial', PolynomialFeatures(deg)),
        #                         ('Regressor', regressor)])
        polyreg.fit(X_train, y_train)
        print(polyreg.named_steps['polynomialfeatures'].n_output_features_)
        y_pred = polyreg.predict(X_val)
        plot_y_yhat(y_val, y_pred)
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        print(f'Degree: {deg} got RMSE of value: {rmse}')
        if rmse < best_error:
            best_model = polyreg

    return best_model, best_error


X, y = GetXandYLists(dataframes)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)
degree = 3
validate_poly_regression(X_train, y_train, X_val, y_val, regressor=LinearRegression())


# plt.figure()
# print(X.shape, y.shape)
# plt.scatter(X,y)
# plt.plot(X,polyreg.predict(X),color="black")
# plt.xlabel('X')
# plt.ylabel('y')