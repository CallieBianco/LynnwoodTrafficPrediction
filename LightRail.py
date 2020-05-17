#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Light Rail class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.15 - 5/13/2020
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

    def avg_day(self, plot=False):
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
        # 35% AM peak - 4 periods
        morn_peak = np.array([.07, .09, .11, .08])
        # 20% mid-day - 5 periods
        mid_day = np.array([.04, .04, .04, .04, .04])
        # 35% PM peak - 4 periods
        aft_peak = np.array([.07, .08, .11, .09])
        # 9% night - 5 periods
        night = np.array([.04, .02, .01, .01, .01])
        trip_probs = np.concatenate((morn, morn_peak, mid_day, 
                                     aft_peak, night))

        # find most accurate number of riders
        anticipated_peak = 2285
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
            plt.suptitle(
                "Randomly Generated Hourly 2024 Light-Rail Passengers \n " + 
                "(Boardings and Alightings)", fontsize=18, x=.51)
            plt.title(
                "Based on Sound Transit Estimates and Current Light-Rail Data")
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

        # want estimate to be within 100 riders
        tol = 100
        best_diffs = []
        best_sizes = []
        sizes = np.arange(start=10000, stop=30000, step=50)
        for s in sizes:
            riders = np.random.choice(times, size=s, p=probs)
            hour_counts = np.bincount(riders)
            # average of trips 6-9am
            am_peak = hour_counts[6:9]
            am_peak = np.mean(am_peak)
            
            # average of trips 3-6pm
            pm_peak = hour_counts[15:18]
            pm_peak = np.mean(pm_peak)

            diff = np.array([abs(peak_est-am_peak), abs(peak_est-pm_peak)])
            c = np.where(diff < tol, 0, diff)
            
            if np.array_equal(c, [0,0]):
                best_diffs.append(np.sum(diff))
                best_sizes.append(s)

        min_i = np.argmin(best_diffs)

        return best_sizes[min_i]

    def get_riders(self, noise=1):
        """
        Generates a week of light-rail riders that will impact traffic
        at 196th based on travel probabilities and busiest traffic days
        
        Returns:
        week_riders: list containing amount of riders for each day M-Sun
        riders_range: list containing a range for expected riders
        """
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
        # find the average of several trials
        trials = np.arange(0, 100, 1)
        ts = np.zeros(len(trials))
        for i in trials:
            ts[i] = self.avg_day()
        riders = int(np.mean(ts))
        l_ride = np.min(ts)
        h_ride = np.max(ts)
        riders_range = [l_ride, h_ride]

        # travel method probabilities given by Sound Transit estimates
        p_car = .21

        # pass 196th probability informed by data
        tot_196 = (45+605+95+50+1100+405+60+585+203+189+381+248)
        prob_196 = ((tot_196 / riders) + p_car)


        # generate day with most traffic
        most_riders = int(riders*1.2)
        most_riders_low = int(l_ride*1.2)
        most_riders_high = int(h_ride*1.2)
        # generate day with least traffic
        least_riders = int(riders*0.8)
        least_riders_low = int(l_ride*1.2)
        least_riders_high = int(h_ride*1.2)

        avg_week_riders = []
        low_week_riders = []
        high_week_riders = []
        week = ['M', 'T', 'W', 'TH', 'F', 'SAT', 'SUN']
        bus_probs = .1
        for day in week:
            impact = prob_196 + bus_probs
            if day == most_traffic:
                impacted = int(most_riders*impact)
                l_impacted = int(most_riders_low*impact)
                h_impacted = int(most_riders_high*impact)
            elif day == least_traffic:
                impacted = int(least_riders*impact)
                l_impacted = int(least_riders_low*impact)
                h_impacted = int(least_riders_high*impact)
            else:
                impacted = int(riders*impact)
                l_impacted = int(l_ride*impact)
                h_impacted = int(h_ride*impact)
            avg_week_riders.append(impacted)
            low_week_riders.append(l_impacted)
            high_week_riders.append(h_impacted)

        return (avg_week_riders, low_week_riders, high_week_riders)

    def impact(self, weekly_riders, low_est_riders, high_est_riders, t2024):
        """
        Using the range of expected average weekly riders, and the
        expected vehicle traffic for 2024, determines the new traffic
        estimate for 2024

        Parameters:
        weekly_riders: int list containing expected average daily ridership
                       for one week
        low_est_riders: int list containing low estimate of expected average
                        daily ridership for one week
        high_est_riders: int list containing high estimate of expected average
                         daily ridership for one week
        t2024: numpy array containing predicted vehicle traffic for one 3-month
               period in 2024
        """

        avg_period = np.repeat(weekly_riders, 13)
        low_period = np.repeat(low_est_riders, 13)
        high_period = np.repeat(high_est_riders, 13)

        avg_2024 = t2024 + period
        low_2024 = t2024 + low_period
        high_2024 = t2024 + high_period