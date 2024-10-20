import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from plot_y_yhat import plot_y_yhat


# GetXandYLists will seperate the training data x and y list
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
                                                time_frame[6],  # x2
                                                
                                                time_frame[9],  # x3
                                                time_frame[10], # y3
                                                
                                                ] 
                    
            else:
                init_state_of_simulation[0] = time_frame[0]  # put current time into the vector instead of the 0
                
                x.append(init_state_of_simulation.copy())
                y.append([time_frame[1], time_frame[2], time_frame[5], time_frame[6], time_frame[9], time_frame[10]])

    return np.array(x), np.array(y)


# we try the polynomial regression from degrees 1 to 10
def validate_poly_regression(X_train, y_train, X_val, y_val,
                             regressor=None, degrees=range(1, 10),
                             max_features=None):
    best_model = None
    best_error = np.inf
    if regressor is None:
        alphas = [0.0001, 0.001, 0.01, 0.1]
        regressor = RidgeCV(alphas=alphas)
    for deg in degrees:
        polyreg = make_pipeline(PolynomialFeatures(deg),
                                StandardScaler(),
                                regressor)
        polyreg.fit(X_train, y_train)

        # predict on training and validation dataset
        y_pred_val = polyreg.predict(X_val)
        y_pred_train = polyreg.predict(X_train)
        # plot_y_yhat(y_val, y_pred_val)

        # compare RMSE 
        rmse_val = math.sqrt(np.square(np.subtract(y_val, y_pred_val)).mean())
        rmse_train = math.sqrt(np.square(np.subtract(y_train, y_pred_train)).mean())

        # Print the number of polynomial features and RMSE for both train and validation
        print(f'Degree {deg}: Created {polyreg.named_steps["polynomialfeatures"].n_output_features_} features.')
        print(f'Degree {deg}: Train RMSE = {rmse_train:.4f}, Validation RMSE = {rmse_val:.4f}')
        
        # Check if this is the best model so far (based on validation RMSE)
        if rmse_val < best_error:
            best_error = rmse_val
            best_model = polyreg

    return best_model, best_error


data = np.load("X_train.npy", allow_pickle=True)
np.random.shuffle(data)
data1pct = data[:int(len(data) * 0.01)]

X, y = GetXandYLists(data1pct)
print(len(X[0]), len(y[0]))
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)
# model, error = validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None)
# print(f'best model{model} and error{error}')


# run the regression with dregree 7 and get the prediction
################################################################
def final_model (X_train, y_train, x_test, regressor=None, degrees=5, max_features=None):

    if regressor is None:
        alphas = [0.0001, 0.001, 0.01, 0.1]
        regressor = RidgeCV(alphas=alphas)
    
    polyreg = make_pipeline(PolynomialFeatures(5), StandardScaler(),regressor)
    polyreg.fit(X_train, y_train)

    y_pred_test = polyreg.predict(x_test)
    
    return y_pred_test

# # prepare the x_test data
# x_test = pd.read_csv("X_test.csv")
# x_test = x_test.drop(['Id'], axis=1)
# x_test = np.save("X_test.npy", x_test)
x_test = np.load("X_test.npy", allow_pickle=True)
# x_test = x_test[:int(len(x_test) * 0.5)]

# use polynomial regression on x_trst dataset
y_pred_test = final_model(X_train, y_train, x_test, regressor=None)
# print(y_pred_test[:3], smalltrain[:1])

# add id and save result
ids = np.arange(0, y_pred_test.shape[0]) 
ids = ids.astype(np.int32)
y_pred = np.column_stack((ids, y_pred_test))  # Use np.column_stack instead of np.hstack

y_pred_df = pd.DataFrame(y_pred, columns=["Id", "x_1", "y_1", "x_2", "y_2", "x_3", "y_3"])
print(y_pred_test.shape)
y_pred_df['Id'] = y_pred_df['Id'].astype(np.int32)  # Make sure 'Id' column is int32

# y_pred_df.to_csv("reduced_polynomial_submission.csv", index=False)

