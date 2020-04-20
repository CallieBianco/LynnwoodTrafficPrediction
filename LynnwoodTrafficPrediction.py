
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
# Version: 1.2 - 4/20/2020
# Written for Python 3.7.2

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

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
    tdf.set_index("Time", inplace=True)
    intersections[i] = tdf
    i+=1

(t196_19, t200_19, t196_18, t200_18) = intersections

def visualize(da, dr):
    """
    Provides a variety of data visualizations
    Parameters:
    da: numpy array of daily totals
    dr: Datetime ndarray storing dates to be plotted
    """
    fig, ax = plt.subplots()
    ax.plot(dr, da, '-oc')

    ax.set(xlabel="Date", ylabel='Total Daily Cars',
       title='Total Daily Cars at 196th and 44th Sept-Nov 2019'
       )
    ax.grid()
    plt.show()

#rng = pd.date_range('1/1/2011', periods=100, freq='1Day')
#print(rng)

def day_sum(df, dates):
    """
    Calculates the daily totals using the hourly totals
    Parameters:
    df: DataFrame
    dates: Datetime ndarray storing the dates to be analyzed

    daily: Numpy array of daily totals
    """
    daily = np.zeros((len(dates),1))
    # sums up the hourly intersections totals by day to get daily total
    daily = df.groupby(df.index.date).sum()
    i = 0
    return daily
    
def tests(df):
    """
    Tests data frame accuracy in a variety of ways
    """
      

def plots():
    #drange19 = pd.date_range(start='9/2/2019', end='11/30/2019')
    #drange18 = pd.date_range(start='9/2/2018', end='11/30/2018')
    #day196_19 = day_sum(t196_19, drange19)
    #day196_19 = day196_19[1:-1]
    #day200_19 = day_sum(t200_19, drange19)
    #day200_19 = day200_19[1:-1]
    #day196_18 = day_sum(t196_18, drange18)
    #day200_18 = day_sum(t196_18, drange18)
    #print(day196_19)
    #visualize(day196_19, drange19)
