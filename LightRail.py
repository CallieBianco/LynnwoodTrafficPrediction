#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Light Rail class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.13 - 5/10/2020
# Written for Python 3.7.2
#==============================================================================

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random as rand
from statistics import mode
from DataInit import DataInitialization
from TES import HoltWinters

class LightRail:
    """
    This class will randomly generate light-rail ridership based on ridership
    estimates from Sound Transit. It will also predict how this ridership
    might affect forecasted vehicle traffic.
    """
    def __init__(self):
        pass

    def weekday(self, plt):
        """
        Generates random weekday light-rail usage 

        Parameters:
        plt: Boolean to plot a histogram of hourly ridership

        Returns:
        riders: array of daily Light-Rail ridership
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

        # find most accurate number of riders
        anticipated_peak = 2500
        riders = np.random.choice(choices, size= 20000, p=trip_probs)
        times = np.sort(riders)

        if plt == True:
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
        return riders
    
    def weekend(self, plt):
        riders = np.random.normal(size=20000)

        if plt == True:
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
        return riders

    def road_impact(self):
        """
        hmm
        """
        # travel probabilities
        p_bus = .66
        p_drive = .19
        p_dropoff = .2
        p_other = .13
        
        # best and worst day factor
        c = DataInitialization()
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
        roads = [t196_19, t196_18, t196_17, t200_19, t200_18, t200_17]
        best = []
        worst = []

        for t in roads:
            (high, low) = c.busy_days(t)
            best.append(high)
            worst.append(low)
        most_traffic = mode(best)
        least_traffic = mode(worst)

        hw = HoltWinters()
        f = hw.forecast_2026()
        print(f)
        return
