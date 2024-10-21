import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
import tensorflow as tf
from scipy import stats
from sklearn.metrics import mean_squared_error
import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.pipeline import make_pipeline
import math

raw_train_data = pd.read_csv("X_train.csv")

raw_train_data['Observation_Id'] = np.repeat(np.arange(5000), 257)
new_array = raw_train_data.to_numpy(dtype=np.float32)
raw_train_data_array = new_array.reshape((5000, 257, 15))

avg1_raw = [[]]
avg2_raw = [[]]
avg3_raw = [[]]

avg1_x_raw = [] 
avg1_y_raw = [] 
avg2_x_raw = [] 
avg2_y_raw = [] 
avg3_x_raw = [] 
avg3_y_raw = []


for single_obs in range(raw_train_data_array.shape[0]):
    obs = pd.DataFrame(raw_train_data_array[single_obs])

    avg1_raw.append([obs.iloc[:, 3].mean(), obs.iloc[:, 4].mean()])
    avg2_raw.append([obs.iloc[:, 7].mean(), obs.iloc[:, 8].mean()])
    avg3_raw.append([obs.iloc[:, 11].mean(), obs.iloc[:, 12].mean()])

    avg1_x_raw.append(obs.iloc[:, 3].mean())
    avg1_y_raw.append(obs.iloc[:, 4].mean())
    avg2_x_raw.append(obs.iloc[:, 7].mean())
    avg2_y_raw.append(obs.iloc[:, 8].mean())
    avg3_x_raw.append(obs.iloc[:, 11].mean())
    avg3_y_raw.append(obs.iloc[:, 12].mean())


avg1_raw.pop(0)
avg2_raw.pop(0)
avg3_raw.pop(0)

avg1_x_raw.pop(0)
avg1_y_raw.pop(0)
avg2_x_raw.pop(0)
avg2_y_raw.pop(0)
avg3_x_raw.pop(0)
avg3_y_raw.pop(0)

def find_empty(in_list):
    id = 0
    ids_empty = []
    for element in range(len(in_list)):
        if in_list[element][0] == 0 and in_list[element][1] == 0:  
            ids_empty.append(id)
        id += 1 
    return(ids_empty)

def count_diff(list_1, list_2, list_3):
    count_diff = 0
    for el in range(len(max(list_1, list_2, list_3))):
        if list_1[el] != list_2[el] or list_1[el] != list_3[el] or list_2[el] != list_3[el]:
            count_diff += 1
    return(count_diff)  


train_data_cleaned_s1 = np.delete(raw_train_data_array, find_empty(avg1_raw), axis=0)

obs_list = []
for i in range(train_data_cleaned_s1.shape[0]):
    obs = train_data_cleaned_s1[i]

    obs_list.append(train_data_cleaned_s1[i][np.any(obs[:, :-2] != 0.0, axis=1)])
train_data_cleaned_s2 = np.array(obs_list, dtype=object)

avg1 = [[]]
avg2 = [[]]
avg3 = [[]]

avg1_x = [] 
avg1_y = [] 
avg2_x = [] 
avg2_y = [] 
avg3_x = [] 
avg3_y = []

z1 = [[]]
z2 = [[]]
z3 = [[]]

std1 = [[]]
std2 = [[]]
std3 = [[]]



for single_obs in range(train_data_cleaned_s2.shape[0]):
    obs = pd.DataFrame(train_data_cleaned_s2[single_obs])

    avg1.append([obs.iloc[:, 3].mean(), obs.iloc[:, 4].mean()])
    avg2.append([obs.iloc[:, 7].mean(), obs.iloc[:, 8].mean()])
    avg3.append([obs.iloc[:, 11].mean(), obs.iloc[:, 12].mean()])

    avg1_x.append(obs.iloc[:, 3].mean())
    avg1_y.append(obs.iloc[:, 4].mean())
    avg2_x.append(obs.iloc[:, 7].mean())
    avg2_y.append(obs.iloc[:, 8].mean())
    avg3_x.append(obs.iloc[:, 11].mean())
    avg3_y.append(obs.iloc[:, 12].mean())

    std1.append([obs.iloc[:, 3].std(), obs.iloc[:, 4].std()])
    std2.append([obs.iloc[:, 7].std(), obs.iloc[:, 8].std()])
    std3.append([obs.iloc[:, 11].std(), obs.iloc[:, 12].std()])
    
    z1.append([stats.zscore(obs.iloc[:, 3]).tolist(), stats.zscore(obs.iloc[:, 4]).tolist()])
    z2.append([stats.zscore(obs.iloc[:, 7]).tolist(), stats.zscore(obs.iloc[:, 8]).tolist()])
    z3.append([stats.zscore(obs.iloc[:, 11]).tolist(), stats.zscore(obs.iloc[:, 12]).tolist()])

avg1.pop(0)
avg2.pop(0)
avg3.pop(0)

avg1_x.pop(0)
avg1_y.pop(0)
avg2_x.pop(0)
avg2_y.pop(0)
avg3_x.pop(0)
avg3_y.pop(0)

std1.pop(0)
std2.pop(0)
std3.pop(0)

z1.pop(0)
z2.pop(0)
z3.pop(0)

def flat_list(list):
    flat_list = []
    for el in list:
        flat_list.extend(el)
    return flat_list    

def compute_norm(dataset):
    x_1, y_1 = [], []
    x_2, y_2 = [], []
    x_3, y_3 = [], []
    for single_obs in range(dataset.shape[0]):
        obs = pd.DataFrame(dataset[single_obs])
        x_1.append(obs.iloc[:, 1])
        y_1.append(obs.iloc[:, 2])
        x_2.append(obs.iloc[:, 5])
        y_2.append(obs.iloc[:, 6])
        x_3.append(obs.iloc[:, 9])
        y_3.append(obs.iloc[:, 10])

    norm_1_2, norm_1_3, norm_2_3 = [], [], []
    for single_obs in range(dataset.shape[0]):
        obs = pd.DataFrame(dataset[single_obs])
        s_list_norm_1_2, s_list_norm_1_3, s_list_norm_2_3 = [], [], []
            
        for moment in range(obs.shape[0]):
            s_list_norm_1_2.append(math.sqrt(math.pow(x_1[single_obs][moment] - x_2[single_obs][moment], 2) + math.pow(y_1[single_obs][moment] - y_2[single_obs][moment], 2)))
            s_list_norm_1_3.append(math.sqrt(math.pow(x_1[single_obs][moment] - x_3[single_obs][moment], 2) + math.pow(y_1[single_obs][moment] - y_3[single_obs][moment], 2)))
            s_list_norm_2_3.append(math.sqrt(math.pow(x_2[single_obs][moment] - x_3[single_obs][moment], 2) + math.pow(y_2[single_obs][moment] - y_3[single_obs][moment], 2)))
        norm_1_2.append(s_list_norm_1_2)
        norm_1_3.append(s_list_norm_1_3)
        norm_2_3.append(s_list_norm_2_3)
    return norm_1_2, norm_1_3, norm_2_3

norm_1_2, norm_1_3, norm_2_3 = compute_norm(train_data_cleaned_s2)

norm_1_2_flatten = flat_list(norm_1_2)
norm_1_3_flatten = flat_list(norm_1_3)
norm_2_3_flatten = flat_list(norm_2_3)

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
                                                
                                                time_frame[15],
                                                time_frame[16],
                                                time_frame[17],
                                                ] 
                    
            else:
                init_state_of_simulation[0] = time_frame[0]  # put current time into the vector instead of the 0
                
                x.append(init_state_of_simulation.copy())
                y.append([time_frame[1], time_frame[2], time_frame[5], time_frame[6], time_frame[9], time_frame[10]])

    return np.array(x), np.array(y)

obs_list = []
for single_obs in range(train_data_cleaned_s2.shape[0]):
        obs = train_data_cleaned_s2[single_obs]
        norm_1_2_col = np.array(norm_1_2[single_obs]).reshape(-1,1)
        norm_2_3_col = np.array(norm_2_3[single_obs]).reshape(-1,1)
        norm_1_3_col = np.array(norm_1_3[single_obs]).reshape(-1,1)
        position = obs.shape[1] - 2
        obs_au = np.hstack([obs[:, :position], np.hstack([norm_1_2_col, norm_2_3_col, norm_1_3_col]), obs[:, position:]])
        obs_list.append(obs_au)
augmented_dataset_1 = np.array(obs_list, dtype=object)  

x_train, y_train = GetXandYLists(augmented_dataset_1)

print(x_train.shape)
