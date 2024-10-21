import numpy as np
import pandas as pd
import math



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

def flat_list(list):
    flat_list = []
    for el in list:
        flat_list.extend(el)
    return flat_list    

data = np.load("X_train.npy", allow_pickle=True)

test_dataset_1 = data
norm_1_2, norm_1_3, norm_2_3 = compute_norm(test_dataset_1)
norm_1_2_flatten = flat_list(norm_1_2)
norm_1_3_flatten = flat_list(norm_1_3)
norm_2_3_flatten = flat_list(norm_2_3)

data_cleaned_list1 = [pd.DataFrame(obs) for obs in test_dataset_1]
augmented_dataframe_1 = pd.concat(data_cleaned_list1, ignore_index=True)

augmented_dataframe_1.insert(len(augmented_dataframe_1.columns) - 2, 'Norm_1_2', norm_1_2_flatten)
augmented_dataframe_1.insert(len(augmented_dataframe_1.columns) - 2, 'Norm_2_3', norm_2_3_flatten)
augmented_dataframe_1.insert(len(augmented_dataframe_1.columns) - 2, 'Norm_1_3', norm_1_3_flatten)

augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[1]: 'x_1'}, inplace=True)
augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[2]: 'y_1'}, inplace=True)
augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[5]: 'x_2'}, inplace=True)
augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[6]: 'y_2'}, inplace=True)
augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[9]: 'x_3'}, inplace=True)
augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[10]: 'y_3'}, inplace=True)

# augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[17]: 'Obs_Id'}, inplace=True)
augmented_dataframe_1.rename(columns={augmented_dataframe_1.columns[16]: 'Id'}, inplace=True)

np.save("augmented_dataframe_1.npy", augmented_dataframe_1)