
#============================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# A project predicting and simulating traffic at two main
# intersections in Lynnwood: 196th/44th and 200th/44th.
# Will predict and analyze traffic impacts before and after
# addition of Link Light-Rail in 2024
#
# Additional Documentation
# Author: Callie Bianco
# Version: 1.0 - 4/15/2020
# Written for Python 3.7.2

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import csv files
t196_19, t200_19, t196_18, t200_18 = None, None, None, None
csv = ["196th_2019.csv", "200th_2019.csv", "196th_2018.csv", "200th_2018.csv"]
intersections = [t196_19, t200_19, t196_18, t200_18]
i = 0

# store csv files in respective intersection object
# format: datetime, int
for file in csv:
    df = pd.read_csv(file, delimiter = ",", 
                     usecols = ["Time", "Int Total"])
    tdf = pd.DataFrame(data = df)
    tdf.drop(tdf.tail(4).index, inplace = True)
    tdf["Time"] = pd.to_datetime(tdf["Time"], utc=True)
    intersections[i] = tdf
    i+=1

(t196_19, t200_19, t196_18, t200_18) = intersections

def visualize(df):
    """
    Provides a variety of data visualizations
    Parameters:
    df: DataFrame
    """
    x = t196_19["Time"]
    y = t196_19["Int Total"]
    plt.plot(x,y)
    plt.show()

#rng = pd.date_range('1/1/2011', periods=100, freq='1Day')
#print(rng)

def day_sum(df, daily):
    """
    Calculates the daily totals using the hourly totals
    Parameters:
    df: DataFrame

    Returns:
    dates: Datetime ndarray storing the dates to be analyzed
    daily: Numpy array of daily totals
    """
    dates = pd.date_range(start='9/1/2019', end='11/30/2019')
    np.zeros((len(dates),1))
    i = 0
    for d in dates:
       y = df.loc[df["Time"].dt.date == d, "Int Total"].sum()
       daily[i] = y
       i+=1 
  
