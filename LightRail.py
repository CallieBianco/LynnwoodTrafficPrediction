#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Light Rail class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.20 - 5/26/2020
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

    def main(self, type=None):
        """
        Main program to be run in the class. 

        Parameters:
        type = char representing which main function to be executed.
            A: plots the histogram of expected avg daily ridership
            B: graphs sensitivity analysis for bus probability variable
            C: plots the impact on vehicle traffic 2024 from light-rail
        """
        # daily ridership
        # A
        if type == "A":
            self.avg_day(plot=True)

        # graph bus probability sensitivity
        # B
        elif type == "B":
            (affecting, low, high) = self.get_riders(bus=True)
            plt.plot(low, label="Low Estimate")
            plt.plot(affecting, label= "Average Estimate")
            plt.plot(high, label= "High Estimate")
            plt.legend()
            plt.xlabel("Varying Bus Probabilities")
            tix = ['.05', '.10', '.15', '.20', '.25', '.30', '.35', '.40', '.45']
            tx = np.arange(0, 70, 7)
            plt.xticks(tx, tix)
            plt.ylabel("Number of Riders Using 196th and 44th")
            plt.title("How Changing Bus Probability Impacts Predicted \n Additional " +
                      "Vehicle Traffic From Light-Rail Riders")
            plt.show()
        
        # traffic impacts
        # C
        elif type == "C":
            hw = HoltWinters()
            (future, model) = hw.forecast_2024(slp=5.1)
            (affecting, low, high) = self.get_riders(bus=False)
            self.impact(affecting, low, high, future, plot=True)
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
        peak_adj = anticipated_peak*.2
        low_peak = anticipated_peak - peak_adj
        high_peak = anticipated_peak + peak_adj

        avg_num_riders = self.rider_estimate(choices, trip_probs, anticipated_peak)
        low_num_riders = self.rider_estimate(choices, trip_probs, low_peak)
        high_num_riders = self.rider_estimate(choices, trip_probs, high_peak)

        avg_riders = np.random.choice(choices, size=avg_num_riders, p=trip_probs)
        low_riders = np.random.choice(choices, size=low_num_riders, p=trip_probs)
        high_riders = np.random.choice(choices, size=high_num_riders, p=trip_probs)
        times = np.sort(avg_riders)

        if plot == True:
            plt.hist(times, bins=80)
        
            # label axis appropriately
            hours = choices.astype('str')
            h = 0
            for h in range(len(hours)):
                hours[h] += ":00"
            plt.suptitle(
                "Randomly Generated Average Hourly 2024 Light-Rail Passengers \n " + 
                "(Boardings and Alightings)", fontsize=18, x=.51)
            plt.title(
                "Based on Sound Transit Estimates and Current Light-Rail Data")
            plt.xlabel("Time")
            plt.ylabel("Hourly Passengers")
            plt.xticks(ticks=choices, labels=hours)
            n_ride = "Number of Riders: " + str(avg_num_riders)
            plt.text(0, 2800, n_ride, bbox=dict(facecolor='green', alpha=.5))
            plt.show()
        return (avg_num_riders, low_num_riders, high_num_riders)
    
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
        sizes = np.arange(start=5000, stop=40000, step=50)
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

    def get_riders(self, bus=False):
        """
        Generates a week of light-rail riders that will impact traffic
        at 196th based on travel probabilities and busiest traffic days
        
        Returns:
        avg_week_riders: list containing average amount of riders for each day M-Sun
        low_week_riders: list containing low end of estimate for amount of riders
        high_week_riders: list containing high end of estimate for amount of riders
        
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
        trials = np.arange(0, 10, 1)
        ars = np.zeros(len(trials))
        lrs = np.zeros(len(trials))
        hrs = np.zeros(len(trials))

        for i in trials:
            (ars[i], lrs[i], hrs[i]) = self.avg_day()
        avg_riders = int(np.mean(ars))
        l_ride = int(np.mean(lrs))
        h_ride = int(np.mean(hrs))

        riders_range = [l_ride, h_ride]

        # travel method probabilities given by Sound Transit estimates
        p_car = .21

        # pass 196th probability informed by data
        tot_196 = (45+605+95+50+1100+405+60+585+203+189+381+248)
        prob_196 = ((tot_196 / avg_riders) + p_car)


        # generate day with most traffic
        most_riders = int(avg_riders*1.2)
        most_riders_low = int(l_ride*1.2)
        most_riders_high = int(h_ride*1.2)

        # generate day with least traffic
        least_riders = int(avg_riders*0.8)
        least_riders_low = int(l_ride*0.8)
        least_riders_high = int(h_ride*0.8)

        avg_week_riders = []
        low_week_riders = []
        high_week_riders = []
        week = ['M', 'T', 'W', 'TH', 'F', 'SAT', 'SUN']
        if bus == True:
            bus_probs = np.arange(start=.05, stop=.5, step=.05)
        else:
            bus_probs = np.arange(start=.25, stop=.3, step=.05)

        for b in bus_probs:
            for day in week:
                impact = prob_196 + b
                if day == most_traffic:
                    impacted = int(most_riders*impact)
                    l_impacted = int(most_riders_low*impact)
                    h_impacted = int(most_riders_high*impact)
                elif day == least_traffic:
                    impacted = int(least_riders*impact)
                    l_impacted = int(least_riders_low*impact)
                    h_impacted = int(least_riders_high*impact)
                else:
                    impacted = int(avg_riders*impact)
                    l_impacted = int(l_ride*impact)
                    h_impacted = int(h_ride*impact)
                avg_week_riders.append(impacted)
                low_week_riders.append(l_impacted)
                high_week_riders.append(h_impacted)

        return (avg_week_riders, low_week_riders, high_week_riders)

    def impact(self, weekly_riders, low_est_riders, high_est_riders, t2024, plot=False):
        """
        Using the range of expected average weekly riders, and the
        expected vehicle traffic for 2024, determines and plots the new 
        traffic estimate for 2024

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

        # add light-rail ridership to previous predicted vehicle counts
        avg_period = np.repeat(weekly_riders, 13)
        low_period = np.repeat(low_est_riders, 13)
        high_period = np.repeat(high_est_riders, 13)

        avg_2024 = t2024 + avg_period
        low_2024 = t2024 + low_period
        high_2024 = t2024 + high_period

        # calculate percentage increases
        h_inc = abs((1 - (np.sum(t2024)/np.sum(high_2024)))) * 100 
        l_inc = abs((1 - (np.sum(t2024)/np.sum(low_2024)))) * 100
        h_inc = int(h_inc)
        l_inc = int(l_inc)

        # plot
        if plot == True:
            plt.plot(t2024, label="Predicted Without Light-Rail")
            plt.plot(avg_2024, label="Average Expected Estimate After Light-Rail")
            plt.plot(low_2024, label="Low Expected Estimate After Light-Rail")
            plt.plot(high_2024, label="High Expected Estimate After Light-Rail")
            plt.legend()
            plt.suptitle("Light-Rail Ridership Predicted Impact on Vehicle Traffic 2024",
                         fontsize=18, x=.51)
            plt.xlabel("Day (Sept.-Nov.)")
            plt.ylabel("Vehicle Count")
            inc_statement = "Predicted Increase In Traffic: " + str(l_inc) + "%-" + str(h_inc) + "%"
            plt.title(inc_statement)
            plt.show()

        return avg_2024, low_2024, high_2024

    def revenue(self, avg_est, low_est, high_est):
        """
        Analyzes revenue in a variety of ways

        Parameters:
        avg_est: int representing average daily riders (avg estimate)
        low_est: int representing average daily riders (low estimate)
        high_est: int representing average daily riders (high estimate)
        """
        # prices
        base_fare = 2.25
        addtl_mile = .05
        exisiting_max = 1
        
        # mileage to stations from city center
        mtlake_terr = 3
        shore_185 = 5
        shore_145 = 7
        northgate = 9
        uw = 15

        min_trip = base_fare + (addtl_mile*mtlake_terr)
        max_trip = base_fare + (addtl_mile*uw) + exisiting_max

        ridership = [low_est, avg_est, high_est]

        # analyze if all riders pay either min or max fare
        all_min_trips = np.zeros(len(ridership))
        all_max_trips = np.zeros(len(ridership))
        
        for i in range(len(ridership)):
            all_min_trips[i] = ridership[i] * min_trip
            all_max_trips[i] = ridership[i] * max_trip

        # plot in grouped bar chart
        labels = ['Minimum Price All Trips', 'Maximum Price All Trips']
        width= .1
        x = np.arange(len(labels))

        low = [all_min_trips[0], all_max_trips[0]]
        avg = [all_min_trips[1], all_max_trips[1]]
        high = [all_min_trips[2], all_max_trips[2]]

        fig, ax = plt.subplots()

        ax.bar(x - width, low, width, label="Low Ridership Estimate")
        ax.bar(x, avg, width, label="Average Ridership Estimate")
        ax.bar(x + width, high, width, label="High Ridership Estimate")

        ax.legend()
        ax.set_ylabel("Revenue ($)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Daily Revenue Generated for Each Ridership Estimate")
        plt.show()
            