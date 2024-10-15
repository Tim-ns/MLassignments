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
    for single_obs in range(data.shape[0]):
        obs = pd.DataFrame(data[single_obs])

        # Averages for the velocity columns
        avg1_raw.append([obs.iloc[:, 3].mean(), obs.iloc[:, 4].mean()])  # for v_x_1, v_y_1
        avg2_raw.append([obs.iloc[:, 7].mean(), obs.iloc[:, 8].mean()])  # for v_x_2, v_y_2
        avg3_raw.append([obs.iloc[:, 11].mean(), obs.iloc[:, 12].mean()])  # for v_x_3, v_y_3
    
    return avg1_raw, avg2_raw, avg3_raw

# Find and remove observations with zero mean
def find_empty(in_list):
    ids_empty = [idx for idx, val in enumerate(in_list) if val[0] == 0 and val[1] == 0]
    return ids_empty

# Count the differences
def count_diff(list_1, list_2, list_3):
    all_elements = list(set(list_1 + list_2 + list_3))  # Combine the unique elements
    count_diff = 0
    for el in all_elements:
        if (el not in list_1) or (el not in list_2) or (el not in list_3):
            count_diff += 1
    return count_diff 

# Remove all rows with zero mean
def remove_0_column(data):
    avg1_raw, avg2_raw, avg3_raw = get_rawagv(data)
    ids_empty = find_empty(avg1_raw) + find_empty(avg2_raw) + find_empty(avg3_raw)
    data_cleaned = np.delete(data, ids_empty, axis=0)
    
    return data_cleaned


# Recalculate mean and std
def newagv(train_data_cleaned_s2):
    avg1, avg2, avg3 = [], [], []
    std1, std2, std3 = [], [], []

    for single_obs in range(train_data_cleaned_s2.shape[0]):
        obs = pd.DataFrame(train_data_cleaned_s2[single_obs])

        avg1.append([obs.iloc[:, 3].mean(), obs.iloc[:, 4].mean()])  # v_x_1, v_y_1
        avg2.append([obs.iloc[:, 7].mean(), obs.iloc[:, 8].mean()])  # v_x_2, v_y_2
        avg3.append([obs.iloc[:, 11].mean(), obs.iloc[:, 12].mean()])  # v_x_3, v_y_3

        std1.append([obs.iloc[:, 3].std(), obs.iloc[:, 4].std()])  # v_x_1, v_y_1
        std2.append([obs.iloc[:, 7].std(), obs.iloc[:, 8].std()])  # v_x_2, v_y_2
        std3.append([obs.iloc[:, 11].std(), obs.iloc[:, 12].std()])  # v_x_3, v_y_3

    for i, avg in enumerate([avg1, avg2, avg3], 1):
        print(f"New Mean for <V_{i}>: {avg[:5]}")  # Just printing first 5 for brevity

    for i, std in enumerate([std1, std2, std3], 1):
        print(f"New Standard Deviation for <V_{i}>: {std[:5]}")  # Just printing first 5 for brevity

    return avg1, avg2, avg3, std1, std2, std3


# Print the number of observations with zero means
def print_agv(data):
    avg1_raw, avg2_raw, avg3_raw = get_rawagv(data)

    for i, avg_raw in enumerate([avg1_raw, avg2_raw, avg3_raw], 1):
        print(f"Number of observations with <V_{i}> having zero mean: {len(find_empty(avg_raw))}")
    
    print("\nLists of <V_1>, <V_2>, <V_3> differ by: ", count_diff(find_empty(avg1_raw), find_empty(avg2_raw), find_empty(avg3_raw)), " elements.")
    return


# Read data
data = readdata("X_train.csv")
print("Original data shape:", data.shape)

# Print the number of observations with zero means
print_agv(data)

# Remove observations with zero mean
data_cleaned = remove_0_column(data)
print("Cleaned data shape:", data_cleaned.shape)

# Print the new mean and std for each observation
avg1, avg2, avg3, std1, std2, std3 = newagv(data_cleaned)
