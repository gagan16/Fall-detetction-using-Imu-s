
import os
import cv2
import pandas as pd
import numpy as np
import configparser as cp
import matplotlib.pyplot as plt


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


data = pd.read_csv("sales")
data = kalman_filter(data)
data=data.iloc[1:]
data.to_csv('sales2.csv', index=False)