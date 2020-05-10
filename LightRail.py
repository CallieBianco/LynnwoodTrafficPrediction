#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Light Rail class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.12 - 5/9/2020
# Written for Python 3.7.2
#==============================================================================

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random as rand

class LightRail:
    """
    This class will randomly generate light-rail ridership based on ridership
    estimates from Sound Transit. It will also predict how this ridership
    might affect forecasted vehicle traffic.
    """
    def __init__(self):
        pass

    def weekday(self):
        """
        Generates random weekday light-rail usage 
        """
        choices = np.arange(0, 24, 1)
        # 1% in the AM - 6 periods 
        # light-rail is closed 2am-5am
        morn = np.array([.003, .002, .0, .0, .0, .005])
        # 40% AM peak - 4 periods
        morn_peak = np.array([.07, .1, .13, .1])
        # 10% mid-day - 5 periods
        mid_day = np.array([.02, .02, .02, .02, .02])
        # 40% PM peak - 5 periods
        aft_peak = np.array([.04, .06, .1, .12, .08])
        # 9% night - 4 periods
        night = np.array([.04, .025, .015, .01])
        trip_probs = np.concatenate((morn, morn_peak, mid_day, aft_peak, night))
        a = np.random.choice(choices, size= 20000, p=trip_probs)
        times = np.sort(a)
        plt.hist(times, bins=80)
        
        # label axis appropriately
        hours = choices.astype('str')
        h = 0
        for h in range(len(hours)):
            hours[h] += ":00"
        plt.suptitle("Randomly Generated Hourly 2026 Light-Rail Passengers \n " + 
        "(Boardings and Alightings)", fontsize=18, x=.51)
        plt.title("Based on Sound Transit Estimates and Current Light-Rail Data")
        plt.xlabel("Time")
        plt.ylabel("Hourly Passengers")
        plt.xticks(ticks=choices, labels=hours)
        plt.show()

    def gbdt():
        """
        Gradient-Boosting Decision Tree algorithm. This algorithm
        will predict the impact light-rail ridership will have on 196th
        and 200th intersections based on travel method, day of week, and 
        desired trip length
        """
        # calculate average ridership

        # calculate residuals

        # construct decision tree

        # predict target label

        # calculate new residuals

        # repeat for number of estimators

        # use all trees to predict target variable
        return