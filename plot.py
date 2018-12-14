import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from math import sqrt
import


def plotaxis(axis, x, y, title):
    axis.plot(x, y)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    axis.set_xlim([min(x), max(x)])
    axis.grid(True)

def plotactivity(act, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plotaxis(ax0, data['timespace'], data['accx'], 'x-axis')
    plotaxis(ax1, data['timespace'], data['accy'], 'y-axis')
    plotaxis(ax2, data['timespace'], data['accz'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(act)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plotacceleration(act, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    ax=data['accx']*data['accx']
    ay = data['accy'] * data['accy']
    az = data['accz'] * data['accz']
    am = ax + ay + az
    am = am.apply(lambda x: math.sqrt(x))
    plotaxis(ax0, data['timespace'], am, 'acceleration')
   # plotaxis(ax0, data['timespace'], data['accy'], 'gyroscope')

    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(act)
    plt.subplots_adjust(top=0.90)
    plt.show()
column_names = ['accx', 'accy', 'accz', 'gccx', 'gccy', 'gccz', 'activity','timspace']

training_data_df = pd.read_csv("D01_SA16_R01normalizationwith time.csv")

for activity in np.unique(training_data_df):
    subset = training_data_df
    plotacceleration(activity,subset)