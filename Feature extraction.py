import pandas as pd
import glob
import os,csv
from scipy.stats import skew, kurtosis
import cv2
import math
import numpy as np
import configparser as cp
from statsmodels.tsa import stattools

def slot(timestamp,d=500):
    start = 0;
    size = timestamp.count();

    while (start < size):
        end = start + d
        yield start, end
        start = start + int(d / 2)

def getfeatures(axis,start,end):
    sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
    return [
        axis[start:end].mean(),
        axis[start:end].std(),
        axis[start:end].var(),
        axis[start:end].min(),
        axis[start:end].max(),
        skew(axis[start:end]),
        kurtosis(axis[start:end]),
        math.sqrt(sqd_error.mean())
    ]

def features(data):
    for (start, end) in slot(data['timespace']):
        features = []
        for axis in ['accx', 'accy', 'accz','gccx','gccy','gccz','accmagnitude', 'gccmagnitude']:
            features += getfeatures(data[axis], start, end)
        yield features

def get_activity(filepath):

    activity=filepath[15:]
    activity=activity[:1]
    #print(activity)
    if activity =='D':
        return 0.0
    elif activity=='F':
        return 1.0
    else:
        return 2.0




with open('features500.csv', 'w') as out:
    rows = csv.writer(out)
   
    path = "Data/magnitude/"
    allFiles = glob.glob(path + "*.csv")
    for file_ in allFiles:
        activity = get_activity(file_)
        for d in features(pd.read_csv(file_)):
            rows.writerow([activity] + d)

