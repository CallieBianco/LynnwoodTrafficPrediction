#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Triple Exponential Smoothing (Holt Winters) algorithm class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.15 - 5/13/2020
# Written for Python 3.7.2
#==============================================================================

# module imports
from DataInit import DataInitialization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random as rand
import math

class HoltWinters:
    """
    This class will create a Holt-Winters methodology for predicting 
    vehicle traffic into the future

    Math guidance credit:
    NIST/SEMATECH e-Handbook of Statistical Methods, 
    https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm
    """
    def __init__(self):
        pass

    def trend(df, season_len, noise=1):
        """
        Calculates the intial trend in the data by taking the average
        of trend averages over the length of each season

        Parameters:
        df: Data Frame containing intersection total values
        season_len: length of a season

        Returns:
        init_trend: float that represents the initial data trend
        """
        sum = 0.0
        for i in range(season_len): 
            sum += (df["Int Total"][i+season_len] 
                    - df["Int Total"][i]) /season_len
        return (sum/season_len) * noise

    def init_season_indices(df, season_len):
        """
        Calculates the initial seasonal indices. Done by dividing each
        value in a season by its seasonal average, then averaging out each
        seasonal length.

        Parameters:
        df: Data Frame containing intersection total values
        season_len: length of one season

        Returns:
        season_indices: initial sesasonal indicies
        """
        # determine number of seasons
        num_days = len(df["Int Total"])
        seasons = int(num_days / season_len)

        # compute average of each season
        int_total = df.to_numpy()
        season_avg = np.reshape(int_total, newshape=(seasons, season_len))
        season_mean = np.mean(season_avg, axis=1)
        
        # repeat average for length of season for division purposes
        season_mean = np.repeat(season_mean, season_len)
        season_mean = np.reshape(season_mean, (seasons, season_len))

        # divide each data point in the season by its seasonal average
        s = (season_avg / season_mean)
        
        # form the seasonal indices
        seasonal_indices = np.mean(s, axis=0)
        return seasonal_indices

    def triple_exp_smooth(self, dfs, season_len, a, b, g, points, rnoise=1):
        """ Uses the inital trend and seasonal inidices to predict
        points in the future

        Parameters:
        dfs: An array containing the dataframes. Each dataframe contains
        traffic data for a different year
        season_len: length of one season
        a: alpha
        b: beta
        g: gamma
        points: how many points in the future I want to forecast to
        """

        # compile yearly traffic counts in one dataframe
        df_new = pd.concat(dfs)
        season_indices = HoltWinters.init_season_indices(df_new, 
                                                         season_len)
        int_total = df_new.to_numpy()
        tot_len = len(int_total)

        # create numpy arrays to store all modeled and forecasted data points
        forecast = np.zeros(tot_len + points)
        future = np.zeros(points)
        j = 0
        # want to predict a certain amount of points past our actual data
        for i in range(tot_len + points):
            #mod_L = i % season_len

            # set initial
            if i == 0:
                smooth = int_total[i]
                sum = 0 
                if tot_len != 364:
                    for t in dfs:
                        sum += HoltWinters.trend(t, season_len)
                    mtrend = sum / (len(dfs) + 1)
                trend = HoltWinters.trend(df_new, season_len, noise=rnoise)
                forecast[i] = int_total[i]
                continue

            # forecast formula:
            # y_x+m = smooth + m*trend + season_indices_x-L+1+(m-1)modL
            if i >= tot_len:
                m = i - tot_len + 1
                s = 0
                if i == tot_len:
                    for t in dfs:
                        s += HoltWinters.trend(t, season_len)
                    mtrend = s / (len(dfs) + 1)
                forecast[i] = (smooth + (m*mtrend) + 
                               (season_indices[i % season_len]))
                future[j] = int((smooth + (m*mtrend) + 
                                 (season_indices[i % season_len])))
                j+=1
            # account for existing points in accordance with model
            else:
                pt = int_total[i]
                prev_smooth = smooth
                smooth = (a * (pt - season_indices[i % season_len]) + 
                          (1 - a) * (smooth + trend))

                prev_trend = trend
                trend = b * (smooth - prev_smooth) + (1 - b) * prev_trend

                season_indices[i % season_len] = (g * (pt - smooth) + 
                                                  (1 - g) * 
                                                  season_indices[i % season_len])
                
                forecast[i] = smooth + trend + season_indices[i % season_len]

        return forecast, future

    def forecast_2024(self, slp, plot=False, n=1):
        """
        Forecasts traffic out to 2026 using a combination of Holt-Winters
        and linear regression

        Parameters:
        slp: regression coefficient
        plot: Default False. If True, provide a plot of the forecasted data
        n: noise level for sensitivity analysis 

        Returns:
        t196_26: Predicted data points for 2024
        """
        p_title = ""
        c = DataInitialization()
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()

        p_title = "196th and 44th Actual/Predicted Traffic 2017-2024"
        veh_counts = [t196_17, t196_18, t196_19]
        t196_21, t196_22, t196_23, t196_24 = 0,0,0,0
        
        # use Holt-Winters to forecast 1-year out
        (modeled, t196_20) = self.triple_exp_smooth(veh_counts, 
                                                    21, .5, .01, 
                                                    .92, 91, rnoise=n)
        
        # forecast until 2026 using Holt Winters forecast and linear
        # regression line
        t196_21 = t196_20 + t196_20*(slp/len(modeled))
        t196_22 = t196_21 + t196_21*(slp/len(modeled))
        t196_23 = t196_22 + t196_22*(slp/len(modeled))
        t196_24 = t196_23 + t196_23*(slp/len(modeled))

        new_roads = [t196_21, t196_22, t196_23, t196_24]
        vs = np.append(modeled, new_roads)

        if plot == True:
            plt.plot(vs, label = "Forecasted")
            plt.title(p_title)
            plt.xlabel("Day Count (September - November)")
            plt.ylabel("Vehicle Count")
            plt.legend()
        return (t196_24, vs)

    def season_avg(self):
        """
        Plots the seasonal averages
        """
        c = DataInitialization()
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
        r_196s = [t196_17, t196_18, t196_19]

        (modeled, t_future) = self.triple_exp_smooth(r_196s, 21, .5, 
                                                     .01, .92, 0)

        # obtain slope line
        x = np.arange(1, len(modeled)+1)
        z = np.polyfit(x, modeled, deg=1)
        p = np.poly1d(z)
        pts = np.arange(1, (len(modeled)*2.7)+1)
        eq = p(pts)
        slope = eq[2] - eq[1]

        # forecast to 2024
        (t24, model) = self.forecast_2024(slp=slope)

        # obtain seasonal averages
        z = np.zeros(7)
        s = np.concatenate((model,z))
        season_avg = np.mean(s.reshape(-1, 21), axis=1)
        season_avg = season_avg[:-1]

        # seasonal trend
        xs = np.arange(1, len(season_avg)+1)
        zs = np.polyfit(xs, season_avg, deg=1)
        ps = np.poly1d(zs)
        ptss = np.arange(1, (len(season_avg))+1)
        plt.plot(ptss, ps(ptss), label = "Trendline")
        
        # plot 
        plt.title("Seasonal Averages (21-Day Season Length)")
        plt.ylabel("Numbers of Vehicles")
        plt.xlabel("Season")
        plt.plot(season_avg, label = "Averages")
        plt.legend()
        plt.show()

    def fitting(self, actual, df1, df2, df3):
        """ 
        Determines the best values of alpha, beta, and gamma to use
        for my Holt-Winters forecasting. Utilizes residual and total sum of
        squares to calculate R^2 and find the optimal parameters.

        Parameters:
        actual: dataframe with the true data
        df1, df2, df3: dataframes to be analyzed in triple_exp_smooth
        """
        alpha = np.arange(start=.5, stop=1, step=.01)
        beta = np.arange(start=.01, stop=.1, step=.01)
        gamma = np.arange(start=.5, stop=1, step=.01)
        avg_err = 0
        std_prop = 0
        for a in alpha:
            for b in beta:
                for g in gamma:
                    (model, future) = self.triple_exp_smooth([df1, df2, df3], 
                                                             21, a, b, g, 0)

                    # average error
                    err = (actual - model)
                    abs_err = abs(err)
                    season_err = np.mean(abs_err.reshape(-1, 21), axis=1)
                    season_avg = np.mean(actual.reshape(-1, 21), axis=1)
                    s_err_prop = (season_err / season_avg)
                    s_err_prop = np.mean(s_err_prop) * 100

                    # standard deviation
                    stdev_m = np.std(model)
                    stdev = np.std(actual)

                    sq_err = err ** 2

                    # calculate residual sum of squares
                    sse = np.sum(sq_err)

                    avg_act = np.mean(actual)
                    sqe = (err - avg_act) ** 2

                    # calculate the total sum of squares
                    sst = np.sum(sqe)

                    r_2 = 1 - sse/sst
                    print("R^2: " + str(r_2))

                    # percent error 
                    std_prop = abs(stdev_m-stdev)
                    std_prop = (std_prop/stdev) * 100
                    print(s_err_prop)
                    print(std_prop)

                    if s_err_prop < 2 and std_prop < 2 and r_2 > .995:
                        print("Alpha: " + str(a))
                        print("Beta: " + str(b))
                        print("Gamma: " + str(g))
                        print("Average Residual Error: " + 
                              str(s_err_prop) + "%")
                        print("Standard Deviation Percent Error: " + 
                              str(std_prop) + "%")
                        return
        
