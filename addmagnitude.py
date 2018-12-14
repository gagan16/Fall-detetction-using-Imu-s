import pandas as pd
import glob
import os
import cv2
import numpy as np
import configparser as cp
import math

def calculatemagnitude():
    path = "Data/kalmanwithtime/"
    allFiles = glob.glob(path + "*.csv")
    for file_ in allFiles:
        data = pd.read_csv(file_)
        ax = data['accx'] * data['accx']
        ay = data['accy'] * data['accy']
        az = data['accz'] * data['accz']
        am = ax + ay + az
        am = am.apply(lambda x: math.sqrt(x))
        data['accmagnitude'] = am
        gx = data['gccx'] * data['gccx']
        gy = data['gccy'] * data['gccy']
        gz = data['gccz'] * data['gccz']
        gm = gx + gy + gz
        gm = gm.apply(lambda x: math.sqrt(x))
        data['gccmagnitude'] = gm
        scaled_training_df = pd.DataFrame(data)
        file_=file_[20:]
        scaled_training_df.to_csv("Data/magnitude/"+file_, index=False)

calculatemagnitude()

