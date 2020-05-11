#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Triple Exponential Smoothing (Holt Winters) algorithm class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.11 - 5/11/2020
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

    def init_season_indices(df, season_len, noise=1):
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
        s = season_avg / season_mean
        
        # form the seasonal indices
        seasonal_indices = ((np.mean(s, axis=0))) * noise
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
                                                         season_len,
                                                         noise=rnoise)
        int_total = df_new.to_numpy()
        tot_len = len(int_total)

        # create numpy arrays to store all modeled and forecasted data points
        forecast = np.zeros(tot_len + points)
        future = np.zeros(points)
        j = 0
        # want to predict a certain amount of points past our actual data
        for i in range(tot_len + points):
            mod_L = i % season_len

            # set initial
            if i == 0:
                smooth = int_total[i]
                sum = 0
                trend = HoltWinters.trend(df_new, season_len, noise=rnoise)
                forecast[i] = int_total[i]
                continue

            # forecast formula:
            # y_x+m = smooth + m*trend + season_indices_x-L+1+(m-1)modL
            if i >= tot_len:
                m = i - tot_len + 1
                s = 0
                mtrend = HoltWinters.trend(df_new, season_len, noise=rnoise)

                # account for yearly population growth
                forecast[i] = (smooth + (m*mtrend) + 
                              (season_indices[mod_L] + 1000))
                future[j] = int((smooth + (m*mtrend) + 
                               (season_indices[mod_L] + 1000)))
                j+=1
            # account for existing points in accordance with model
            else:
                pt = int_total[i]
                prev_smooth = smooth
                smooth = a * (pt - season_indices[mod_L]) + (1 - a) * (smooth + trend)

                trend = b * (smooth - prev_smooth) + (1 - b) * trend

                season_indices[mod_L] = g * (pt - smooth) + (1 - g) * season_indices[mod_L]

                forecast[i] = smooth + trend + season_indices[mod_L]
        return forecast, future

    def forecast_2026(self, plot=False, n=1):
        """
        Forecasts traffic out to 2026
        """
        c = DataInitialization()
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
        # 2020
        veh_counts = [t196_17, t196_18, t196_19]
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 
                                                       91, .54, .02, 
                                                       .86, 91, rnoise=n)
        d2020 = pd.date_range('9-01-2020', periods=91, freq='D')
        new = {'Time': d2020, 'Int Total': future_pts}
        t196_20 = pd.DataFrame(data=new)
        t196_20.set_index("Time", inplace=True)

        # 2021
        veh_counts.append(t196_20)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 
                                                       91, .54, .02, 
                                                       .86, 91, rnoise=n)
        d2021 = pd.date_range('9-01-2021', periods=91, freq='D')
        new = {'Time': d2021, 'Int Total': future_pts}
        t196_21 = pd.DataFrame(data=new)
        t196_21.set_index("Time", inplace=True)
        
        # 2022
        veh_counts.append(t196_21)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 
                                                       91, .54, .02, 
                                                       .86, 91, rnoise=n)
        d2022 = pd.date_range('9-01-2022', periods=91, freq='D')
        new = {'Time': d2022, 'Int Total': future_pts}
        t196_22 = pd.DataFrame(data=new)
        t196_22.set_index("Time", inplace=True)

        # 2023
        veh_counts.append(t196_22)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 
                                                       91, .54, .02, 
                                                       .86, 91, rnoise=n)
        d2023 = pd.date_range('9-01-2023', periods=91, freq='D')
        new = {'Time': d2023, 'Int Total': future_pts}
        t196_23 = pd.DataFrame(data=new)
        t196_23.set_index("Time", inplace=True)

        # 2024
        veh_counts.append(t196_23)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 
                                                       91, .54, .02, 
                                                       .86, 91, rnoise=n)
        d2024 = pd.date_range('9-01-2024', periods=91, freq='D')
        new = {'Time': d2024, 'Int Total': future_pts}
        t196_24 = pd.DataFrame(data=new)
        t196_24.set_index("Time", inplace=True)

        # 2025
        veh_counts.append(t196_24)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 
                                                       91, .54, .02, 
                                                       .86, 91, rnoise=n)
        d2025 = pd.date_range('9-01-2025', periods=91, freq='D')
        new = {'Time': d2025, 'Int Total': future_pts}
        t196_25 = pd.DataFrame(data=new)
        t196_25.set_index("Time", inplace=True)

        # 2026
        veh_counts.append(t196_25)
        (modeled, future) = self.triple_exp_smooth(veh_counts, 
                                                       91, .54, .02, 
                                                       .86, 91, rnoise=n)
        d2026 = pd.date_range('9-01-2026', periods=91, freq='D')
        new = {'Time': d2026, 'Int Total': future_pts}
        t196_26 = pd.DataFrame(data=new)
        t196_26.set_index("Time", inplace=True)
        if plot == True:
            plt.plot(modeled, label = "Forecasted, " + str(n) + " Noise Level")
       
            # want to get trend for actual data
            plt.title("196th and 44th Actual/Predicted Traffic 2017-2026")
            plt.xlabel("Day Count (September - November)")
            plt.ylabel("Vehicle Count")
            plt.legend()
        return future

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
