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
class HoltWinters:
    """
    This class will create a Holt-Winters methodology for predicting 
    vehicle traffic into the future

    Math guidance credit:
    NIST/SEMATECH e-Handbook of Statistical Methods, 
    https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm, Apr 21, 2020
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
                forecast[i] = (smooth + (m*mtrend) + (season_indices[i % season_len]))
                future[j] = int((smooth + (m*mtrend) + (season_indices[i % season_len])))
                j+=1
            # account for existing points in accordance with model
            else:
                pt = int_total[i]
                prev_smooth = smooth
                smooth = a * (pt - season_indices[i % season_len]) + (1 - a) * (smooth + trend)

                prev_trend = trend
                trend = b * (smooth - prev_smooth) + (1 - b) * prev_trend

                season_indices[i % season_len] = g * (pt - smooth) + (1 - g) * season_indices[i % season_len]
                
                forecast[i] = smooth + trend + season_indices[i % season_len]

        return forecast, future

    def forecast_2026(self, slp, plot=False, n=1):
        """
        Forecasts traffic out to 2026 using a combination of Holt-Winters
        and linear regression

        Parameters:
        slp: regression coefficient
        plot: Default False. If True, provide a plot of the forecasted data
        n: noise level for sensitivity analysis 

        Returns:
        t196_26: Predicted data points for 2026
        """
        p_title = ""
        c = DataInitialization()
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()

        p_title = "196th and 44th Actual/Predicted Traffic 2017-2026"
        veh_counts = [t196_17, t196_18, t196_19]
        t196_21, t196_22, t196_23, t196_24, t196_25, t196_26 = 0,0,0,0,0,0
        
        # use Holt-Winters to forecast 1-year out
        (modeled, t196_20) = self.triple_exp_smooth(veh_counts, 
                                                    21, .54, .02, 
                                                    .86, 91, rnoise=n)
        
        # forecast until 2026 using Holt Winters forecast and linear
        # regression line
        t196_21 = t196_20 + t196_20*(slp/len(modeled))
        t196_22 = t196_21 + t196_21*(slp/len(modeled))
        t196_23 = t196_22 + t196_22*(slp/len(modeled))
        t196_24 = t196_23 + t196_23*(slp/len(modeled))
        t196_25 = t196_24 + t196_24*(slp/len(modeled))
        t196_26 = t196_25 + t196_25*(slp/len(modeled))

        new_roads = [t196_21, t196_22, t196_23, t196_24, t196_25, t196_26]
        vs = np.append(modeled, new_roads)

        if plot == True:
            plt.plot(vs, label = "Forecasted")
            plt.title(p_title)
            plt.xlabel("Day Count (September - November)")
            plt.ylabel("Vehicle Count")
            plt.legend()
        return t196_26

def fitting(actual, df1, df2, df3):
    """ 
    Determines the best values of alpha, beta, and gamma to use
    for my Holt-Winters forecasting. Utilizes residual and total sum of
    squares to calculate R^2 and find the optimal parameters.

    Parameters:
    actual: dataframe with the true data
    df1, df2, df3: dataframes to be analyzed in triple_exp_smooth
    """
    alpha = np.arange(start=0, stop=1, step=.02)
    beta = np.arange(start=0, stop=1, step=.02)
    gamma = np.arange(start=0, stop=1, step=.02)
    r_best = -20000
    best = ()
    for a in alpha:
        for b in beta:
            for g in gamma:
                # forecast
                h = hw.triple_exp_smooth([df1, df2, df3], 21, a, b, g, 0)
                sq_err = (d - h)**2

                # calculate residual sum of squares
                sse = np.sum(sq_err)

                avg_act = np.mean(d)
                sqe = (d - avg_act) ** 2

                # calculate the total sum of squares
                sst = np.sum(sqe)

                r_2 = 1 - sse/sst

                # find alpha, beta, and gamma with lowest R^2 value
                if r_2 > r_best:
                    best = (a, b, g)
                    r_best = r_2
                    print(str(r_best))
    return best

#print(fitting(d, t196_17, t196_18, t196_19))
