#==============#============================================================
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
# Version: 1.9 - 5/4/2020
# Written for Python 3.7.2

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random as rand

class DataInitialization:
    """
    This class deals with all the initial cleaning and parsing of 
    the traffic data. It also condenses hourly counts to daily,
    provides statistical analysis of the data, and visualizations
    """
    def __init__(self):
        pass

    def main(self, visual=None):
        """
        Main program to be run in the class
        """
        # provide inidivual plots
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = self.read_files()
        # A
        if visual == "A":
            self.visualize(t196_19, "singleplot", 
                           "Daily Traffic Counts at 196th and 44th - 2019",
                           'c')  
            self.visualize(t196_18, "singleplot", 
                           "Daily Traffic Counts at 196th and 44th - 2018",
                           'g')  
            self.visualize(t200_19, "singleplot", 
                           "Daily Traffic Counts at 200th and 44th - 2019",
                           'r')   
            self.visualize(t200_18, "singleplot", 
                           "Daily Traffic Counts at 200th and 44th - 2018",
                           'k')   
        # combined years
        # B
        elif visual == "B":
            self.visualize(t196_19, "multiplot", 
                           "Daily Traffic Counts at 196th and 44th 2018-2019",
                           'b', df1=t196_18)
            self.visualize(t200_19, "multiplot", 
                           "Daily Traffic Counts at 200th and 44th 2018-2019",
                           'b', df1=t200_18)

        elif visual == "C":
            self.visualize(t196_17, "singleplot", 
                           "Daily Traffic Counts at 196th and 44th - 2017",
                           'b')  
            self.visualize(t200_17, "singleplot", 
                           "Daily Traffic Counts at 200th and 44th - 2017",
                           'g')   
            self.visualize(t196_18, "multiplot", 
                           "Daily Traffic Counts at 200th and 44th 2018-2019",
                           'b', df1=t196_17)

    def read_files(self):
        # import csv files
        t196_19, t200_19, t196_18, t200_18, t196_17, t200_17 = None, None, None, None, None, None
        csv = ["196th_2019.csv", "200th_2019.csv", 
               "196th_2018.csv", "200th_2018.csv",
               "196th_2017.csv", "200th_2017.csv"]
        intersections = [t196_19, t200_19, t196_18, t200_18, t196_17, t200_17]
        i = 0

        # store csv files in respective intersection object
        # format: datetime, int
        for file in csv:
            df = pd.read_csv(file, delimiter = ",", 
                             usecols = ["Time", "Int Total"])
            tdf = pd.DataFrame(data = df)

            # remove the last few rows that break format
            tdf.drop(tdf.tail(4).index, inplace = True)

            # convert "Time" to datetime
            tdf["Time"] = pd.to_datetime(tdf["Time"], utc=True)
            tdf.set_index("Time", inplace=True)
            intersections[i] = tdf
            i+=1

        (t196_19, t200_19, t196_18, t200_18, t196_17, t200_17) = intersections
        # obtain daily counts instead of hourly
        t196_19 = self.day_sum(t196_19)
        t196_18 = self.day_sum(t196_18)
        t196_17 = self.day_sum(t196_17)
        t200_19 = self.day_sum(t200_19)
        t200_18 = self.day_sum(t200_18)
        t200_17 = self.day_sum(t200_17)
        return (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17)
    def day_sum(self, df):
        """
        Calculates the daily totals using the hourly totals
        Parameters:
        df: DataFrame
        Returns:
        daily: Numpy array of daily totals
        """
        daily = df.groupby(df.index.date).sum()
        return daily

    def visualize(self, df, ptype, title, color, df1 = None):
        """
        Provides a variety of data visualizations
        Parameters:
        df: Dataframe to be visualized
        ptype: String with the name of the plot type
        title: String with title for graph
        color: Char for color of graph
        """
        
        compareType = ptype.lower()

        # plotting a single year
        if compareType == "SinglePlot".lower():
            # create a line plot using matplotlib and pandas
           fig, ax = plt.subplots()
           ax.plot(df, color)

           ax.set(xlabel="Date", ylabel='Total Daily Cars',
           title=title)
           ax.grid()
           plt.show()

        # plotting multiple years
        elif compareType == "MultiPlot".lower():
           frames = [df1, df]
           df2 = pd.concat(frames)
           df2["Time"] = df2.index
           df2["Time"] = pd.to_datetime(df2["Time"], utc=True)
           df2["Day"] = df2["Time"].dt.dayofyear
           df2["Year"] = df2["Time"].dt.year
           print(df2)
           piv = pd.pivot_table(df2, index="Day",
                                columns="Year", values=["Int Total"])
           piv.plot()
           plt.ylabel("Total Cars")
           plt.xlabel("Day: Sept. - Nov.")
           plt.title(title)
           plt.show()
        else:
            return "Invalid plot type given"
    
    def tests(self):
        """
        Tests data frame accuracy in a variety of ways
        """     
        # Test 1: confirm 91 days in each dataset after totalling
        print("Beginning testing for Data Initialization Class")
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
        counts = [t196_19, t196_18, t196_17, t200_19, t200_18, t200_17]
        for d in counts:
            if len(d) != 91:
                print("Test 1: Failed.")
                return
        print("Test 1: Passed")

        # Test 2: Correct daily sums. I calculated a few in excel to test
        if t196_17["Int Total"][0] != 62064:
            print("Test 2: Failed")
            return
        if t200_18["Int Total"][0] != 40473:
            print("Test 2: Failed")
            return
        if t196_19["Int Total"][15] != 65060:
            print("Test 2: Failed")
            return
        if t200_19["Int Total"][89] != 47533:
            print("Test 2: Failed")
            return
        print("Test 2: Passed")

        # Test 3: Bad parameter error checking
        message = c.visualize(t196_19, "Histogram", "Plot", 'k')
        if message != "Invalid plot type given":
            print("Test 3: Failed")
            return
        print("Test 3: Passed")
        print("Testing finished. All tests passed.")

def data_reqs():
    """
    Run all code for 'Data' requirements
    """
    # running the program so far
    c = DataInitialization()
    (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
    c.main("C")
    c.tests()

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

    def trend(df, season_len):
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
        return sum/season_len

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
        s = season_avg / season_mean
        
        # form the seasonal indices
        seasonal_indices = np.mean(s, axis=0)
        return seasonal_indices

    def triple_exp_smooth(dfs, season_len, a, b, g, points):
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
        season_indices = HoltWinters.init_season_indices(df_new, season_len)
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
                #if tot_len != 364:
                #    for t in dfs:
                #        sum += HoltWinters.trend(t, season_len)
                #    mtrend = sum / (len(dfs) + 1)
                trend = HoltWinters.trend(df_new, season_len)
                forecast[i] = int_total[i]
                continue

            # forecast formula:
            # y_x+m = smooth + m*trend + season_indices_x-L+1+(m-1)modL
            if i >= tot_len:
                m = i - tot_len + 1
                s = 0
                #if tot_len != 364:
                #    for t in dfs:
                #        s += HoltWinters.trend(t, season_len)
                #    mtrend = s / (len(dfs) + 1)
                mtrend = HoltWinters.trend(df_new, season_len)

                # account for yearly population growth
                # add some randomness
                forecast[i] = (smooth + (m*mtrend) + (season_indices[mod_L] + 1000))
                future[j] = int((smooth + (m*mtrend) + (season_indices[mod_L] + 1000)))
                j+=1
            # account for existing points in accordance with model
            else:
                pt = int_total[i]
                prev_smooth = smooth
                smooth = a * (pt - season_indices[mod_L]) 
                + (1 - a) * (smooth + trend)
                trend = b * (smooth - prev_smooth) + (1 - b) * trend
                season_indices[mod_L] = g * (pt - smooth) + (1 - g) * season_indices[mod_L]
                forecast[i] = smooth + trend + season_indices[mod_L]
        return forecast, future

    def forecast_2026(self):
        """
        Forecasts traffic out to 2026
        """
        c = DataInitialization()
        (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
        # 2020
        veh_counts = [t196_17, t196_18, t196_19]
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 91, .54, .02, .86, 91)
        d2020 = pd.date_range('9-01-2020', periods=91, freq='D')
        new = {'Time': d2020, 'Int Total': future_pts}
        t196_20 = pd.DataFrame(data=new)
        t196_20.set_index("Time", inplace=True)

        # 2021
        veh_counts.append(t196_20)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 91, .54, .02, .86, 91)
        d2021 = pd.date_range('9-01-2021', periods=91, freq='D')
        new = {'Time': d2021, 'Int Total': future_pts}
        t196_21 = pd.DataFrame(data=new)
        t196_21.set_index("Time", inplace=True)
        
        # 2022
        veh_counts.append(t196_21)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 91, .54, .02, .86, 91)
        d2022 = pd.date_range('9-01-2022', periods=91, freq='D')
        new = {'Time': d2022, 'Int Total': future_pts}
        t196_22 = pd.DataFrame(data=new)
        t196_22.set_index("Time", inplace=True)

        # 2023
        veh_counts.append(t196_22)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 91, .54, .02, .86, 91)
        d2023 = pd.date_range('9-01-2023', periods=91, freq='D')
        new = {'Time': d2023, 'Int Total': future_pts}
        t196_23 = pd.DataFrame(data=new)
        t196_23.set_index("Time", inplace=True)

        # 2024
        veh_counts.append(t196_23)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 91, .54, .02, .86, 91)
        d2024 = pd.date_range('9-01-2024', periods=91, freq='D')
        new = {'Time': d2024, 'Int Total': future_pts}
        t196_24 = pd.DataFrame(data=new)
        t196_24.set_index("Time", inplace=True)

        # 2025
        veh_counts.append(t196_24)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 91, .54, .02, .86, 91)
        d2025 = pd.date_range('9-01-2025', periods=91, freq='D')
        new = {'Time': d2025, 'Int Total': future_pts}
        t196_25 = pd.DataFrame(data=new)
        t196_25.set_index("Time", inplace=True)

        # 2026
        veh_counts.append(t196_25)
        (modeled, future_pts) = self.triple_exp_smooth(veh_counts, 91, .76, .03, .91, 91)
        d2026 = pd.date_range('9-01-2026', periods=91, freq='D')
        new = {'Time': d2026, 'Int Total': future_pts}
        t196_26 = pd.DataFrame(data=new)
        t196_26.set_index("Time", inplace=True)
        v = [t196_17, t196_18, t196_19]
        d = pd.concat(v)
        d = d.to_numpy()
        plt.plot(d, label = "Actual (Until 2019)")
        plt.plot(modeled, label = "Forecasted")
       
        # want to get trend for actual data
        x = np.arange(1, len(modeled)+1)
        z = np.polyfit(x, modeled, deg=1)
        p = np.poly1d(z)
        plt.plot(x, p(x), label = "Trendline")
        plt.title("196th and 44th Actual/Predicted Traffic 2017-2026")
        plt.xlabel("Day Count (September - November)")
        plt.ylabel("Vehicle Count")
        plt.legend()
        plt.show()

def TES_reqs():
    hw = HoltWinters
    hw.forecast_2026(hw)
    
TES_reqs()


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
