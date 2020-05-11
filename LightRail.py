#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Light Rail class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.14 - 5/11/2020
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

    def weekday(self, plot=False):
        """
        Generates random weekday light-rail usage 

        Parameters:
        plot: Boolean to plot a histogram of hourly ridership

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
        num_riders = self.rider_estimate(choices, trip_probs, anticipated_peak)
        riders = np.random.choice(choices, size=num_riders, p=trip_probs)
        times = np.sort(riders)

        if plot == True:
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
            n_ride = "Number of Riders: " + str(num_riders)
            plt.text(0, 2800, n_ride, bbox=dict(facecolor='green', alpha=.5))
            plt.show()
        return num_riders
    
    def rider_estimate(self, times, probs, peak_est):
        """
        Finds the most appropriate estimate of daily boardings and alightings 
        based on how well the distribution peak times reflect 
        the expected peak times

        Parameters:
        times: 24 hour cycle of times to choose from
        probs: probabilities for each hour
        peak_est: Integer estimating number of boardings/alightings during 
                  the peak hours (7-11am and 4-7pm)

        Returns:
        rider_size: Integer with appropriate number of daily riders
        """

        # want estimate to be within 50 riders
        tol = 50
        best_diffs = []
        best_sizes = []
        sizes = np.arange(start=10000, stop=30000, step=50)
        for s in sizes:
            riders = np.random.choice(times, size=s, p=probs)
            hour_counts = np.bincount(riders)
            # average of trips 7-9am
            am_peak = hour_counts[7:9]
            am_peak = np.mean(am_peak)
            
            # average of trips 5-7pm
            pm_peak = hour_counts[17:19]
            pm_peak = np.mean(pm_peak)

            diff = np.array([abs(peak_est-am_peak), abs(peak_est-pm_peak)])
            c = np.where(diff < tol, 0, diff)
            
            if np.array_equal(c, [0,0]):
                best_diffs.append(np.sum(diff))
                best_sizes.append(s)

        min_i = np.argmin(best_diffs)

        return best_sizes[min_i]

    def weekend(self, plot):
        choices = np.arange(0, 24, 1)
        riders = np.random.normal(12, 2, size=20000)

        if plot == True:
            plt.hist(riders, bins=80)
        
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
        # travel method probabilities
        # given by Sound Transit estimates
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

        # light-rail daily riders

        riders = self.weekday()

        # create a 3-month period using this average ridership
        
        avg_daily_riders = np.repeat(riders, 91)
        print(avg_daily_riders)
        
        hw = HoltWinters()
        f = hw.forecast_2026()
        print(f)
        return