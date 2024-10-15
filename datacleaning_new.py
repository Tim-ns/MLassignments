import pandas as pd
import numpy as np

# Read the raw data
def readdata(data):
    data = pd.read_csv(data)
    data['Observation_Id'] = np.repeat(np.arange(5000), 257)
    new_array = data.to_numpy(dtype=np.float32)
    raw_train_data_array = new_array.reshape((5000, 257, 15))
    return raw_train_data_array

# Calculate the averages
def get_rawagv(data):

    avg1_raw, avg2_raw, avg3_raw = [], [], []
    avg1_x_raw, avg1_y_raw, avg2_x_raw, avg2_y_raw, avg3_x_raw, avg3_y_raw = [], [], [], [], [], []

    for single_obs in range(data.shape[0]):
        obs = pd.DataFrame(data[single_obs])

        avg1_raw.append([obs.iloc[:, 3].mean(), obs.iloc[:, 4].mean()])
        avg2_raw.append([obs.iloc[:, 7].mean(), obs.iloc[:, 8].mean()])
        avg3_raw.append([obs.iloc[:, 11].mean(), obs.iloc[:, 12].mean()])

        avg1_x_raw.append(obs.iloc[:, 3].mean())
        avg1_y_raw.append(obs.iloc[:, 4].mean())
        avg2_x_raw.append(obs.iloc[:, 7].mean())
        avg2_y_raw.append(obs.iloc[:, 8].mean())
        avg3_x_raw.append(obs.iloc[:, 11].mean())
        avg3_y_raw.append(obs.iloc[:, 12].mean())
    
    return avg1_raw, avg2_raw, avg3_raw, avg1_x_raw, avg1_y_raw, avg2_x_raw, avg2_y_raw, avg3_x_raw, avg3_y_raw


#Find and remove observations with 0 mean
def find_empty(in_list):
    id = 0
    ids_empty = []
    for element in range(len(in_list)):
        if in_list[element][0] == 0 and in_list[element][1] == 0:  
            ids_empty.append(id)
        id += 1 
    return ids_empty


def count_diff(list_1, list_2, list_3):
    count_diff = 0
    for el in range(len(max(list_1, list_2, list_3))):
        if list_1[el] != list_2[el] or list_1[el] != list_3[el] or list_2[el] != list_3[el]:
            count_diff += 1
    return count_diff 

# Remove all rows with zero mean
def remove_0_column(data):
    data = np.delete(data, find_empty(avg1_raw), axis=0)
    obs_list = []
    for i in range(data.shape[0]):
        obs = data[i]

        obs_list.append(data[i][np.any(obs[:, :-2] != 0.0, axis=1)])
    train_data_cleaned_s2 = np.array(obs_list, dtype=object)

    return train_data_cleaned_s2

# Recalculate mean, std for every observation
def newagv(train_data_cleaned_s2):
    avg1, avg2, avg3= [], [], []
    std1, std2, std3 = [], [], []

    for single_obs in range(train_data_cleaned_s2.shape[0]):
        obs = pd.DataFrame(train_data_cleaned_s2[single_obs])

        avg1.append([obs.iloc[:, 3].mean(), obs.iloc[:, 4].mean()])
        avg2.append([obs.iloc[:, 7].mean(), obs.iloc[:, 8].mean()])
        avg3.append([obs.iloc[:, 11].mean(), obs.iloc[:, 12].mean()])

        std1.append([obs.iloc[:, 3].std(), obs.iloc[:, 4].std()])
        std2.append([obs.iloc[:, 7].std(), obs.iloc[:, 8].std()])
        std3.append([obs.iloc[:, 11].std(), obs.iloc[:, 12].std()])
        

    return avg1, avg2, avg3, std1, std2, std3

data = readdata("X_train.csv")
avg1_raw, avg2_raw, avg3_raw, avg1_x_raw, avg1_y_raw, avg2_x_raw, avg2_y_raw, avg3_x_raw, avg3_y_raw = get_rawagv(data)

# Print the number of observations with 0 means
for i, avg_raw in enumerate([avg1_raw, avg2_raw, avg3_raw], 1):
    print(f"Number of observations with <V_{i}>: len(find_empty({avg_raw}))")
print("\nLists of <V_1>, <V_2>, <V_3> differ by: ", count_diff(find_empty(avg1_raw), find_empty(avg2_raw), find_empty(avg3_raw)), " elements.")

# remove observations with zero mean
data_cleaned = remove_0_column(data)
print(data_cleaned.shape)

# Print the new mean, std, for every observation  
avg1, avg2, avg3,std1, std2, std3 = newagv(data_cleaned)
for i, avg in enumerate([avg1, avg2, avg3], 1):
    print(f"New Mean for <V_{i}>: {avg}")

for i, std in enumerate([std1, std2, std3], 1):
    print(f"New Standard Deviation for <V_{i}>: {std}")



