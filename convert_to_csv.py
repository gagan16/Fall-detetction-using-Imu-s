import pandas as pd
import glob
import os
import cv2
import numpy as np
import configparser as cp
import matplotlib.pyplot as plt


def read_txt(file_path):
    column_names = ['accx', 'accy', 'accz', 'gccx', 'gccy', 'gccz', 'acx', 'acy', 'acz']
    data = pd.read_csv(file_path, header=None, names=column_names)
    data = data.replace(';', '.', regex=True)
    data = data.drop('acx', axis=1)
    data = data.drop('acy', axis=1)
    data = data.drop('acz', axis=1)

   # data['answer'] = get_activity(file_path)
    scaled_data = kalman_filter(data)
    scaled_data=scaled_data.iloc[1:]
    column_names2 = ['accx', 'accy', 'accz', 'gccx', 'gccy', 'gccz', ]

    scaled_training_df = pd.DataFrame(scaled_data,columns=column_names2)
    filepath=file_path[14:]
    scaled_training_df.to_csv("Data/kalman/"+filepath[:-4]+".csv", index=False)

# #def feature_normalization(dataset):
#     # network to work well.
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     # Scale both the training inputs and outputs
#     scaled_training = scaler.fit_transform(dataset)
#     return scaled_training

def kalman_filter(data):
    kalman = cv2.KalmanFilter(6, 6)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 0, 0, 1]], np.float32) * 1

    row_num = data.accx.size

    for i in range(row_num):
        correct = np.array(data.iloc[i, 0:6].values, np.float32).reshape([6, 1])
        kalman.correct(correct)
        predict = kalman.predict()
        data.iloc[i, 0] = predict[0]
        data.iloc[i, 1] = predict[1]
        data.iloc[i, 2] = predict[2]
        data.iloc[i, 3] = predict[3]
        data.iloc[i, 4] = predict[4]
        data.iloc[i, 5] = predict[5]

    return data

def converting():
    path = "Data/txt/"
    allFiles = glob.glob(path + "*")
    for file_ in allFiles:
        allfiles2 =glob.glob(file_+"/*.txt")
        for file_2 in allfiles2:
            read_txt(file_2)
            #get_activity(file_2)

converting()