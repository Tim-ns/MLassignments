import numpy as np


def GetXandYLists(data: np.array, account_for_velocity=False):
    x = []
    y = []

    for simulation_matrix in data:
        # print("simulation matrix= ", simulation_matrix)
        init_state_of_simulation = []
        for num, time_frame in enumerate(simulation_matrix):
            # print("time_frame= ", (time_frame[:3]))
            if num == 0:
                if account_for_velocity:
                    init_state_of_simulation = time_frame[:-2]  # remove the last two elements, because they are ids
                else:
                    init_state_of_simulation = [time_frame[0],  # time
                                                time_frame[1],  # x1
                                                time_frame[2],  # y1
                                                time_frame[3],  # x2
                                                time_frame[4],  # y2
                                                time_frame[5],  # x3
                                                time_frame[6], # y3
                                                ] 
                    
            else:
                init_state_of_simulation[0] = time_frame[0]  # put current time into the vector instead of the 0
                
                x.append(init_state_of_simulation.copy())
                y.append([time_frame[1], time_frame[2], time_frame[3], time_frame[4], time_frame[5], time_frame[6]])

    return np.array(x), np.array(y)


data = np.load("X_train.npy", allow_pickle=True)

#testing data
#data = np.array([np.array([np.array(["ti", "x1i", "y1i", "v1xi", "v1yi", "x2i", "y2i", "v2xi", "v2yi", "x3i", "y3i", "v3xi", "v3yi", "idi", "oidi"])
#                , np.array(["t", "x1", "y1", "v1x", "v1y", "x2", "y2", "v2x", "v2y", "x3", "y3", "v3x", "v3y", "id", "oid"])])])

x, y = GetXandYLists(data)
print(len(x[0]), len(y[0]))

