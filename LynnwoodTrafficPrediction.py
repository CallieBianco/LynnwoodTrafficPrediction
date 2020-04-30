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
# Version: 1.8 - 4/29/2020
# Written for Python 3.7.2

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns

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
        da: numpy array of daily totals
        dr: Datetime ndarray storing dates to be plotted
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
            print("Invalid plot type given")
    
    def tests(self):
        """
        Tests data frame accuracy in a variety of ways
        """     
        # confirm each daily total consists of 24 hourly periods
        (t196_19, t196_18, t200_19, t200_18) = self.read_files()
        print(t196_19)
        

# running the program so far
#c = DataInitialization()
#(t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
#c.main("C")

# tests
#c.tests()

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
        season_len: length of a season

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
        """
        df4 = pd.concat(dfs)
        season_indices = HoltWinters.init_season_indices(df4, season_len)
        int_total = df4.to_numpy()
        tot_len = len(int_total)
        forecast = np.zeros(tot_len + points)
        # want to predict a certain amount of points past our actual data
        for i in range(tot_len + points):
            mod_L = i % season_len
            # set initial
            if i == 0:
                smooth = int_total[i]
                sum = 0
                for t in dfs:
                    sum += HoltWinters.trend(t, season_len)
                trend = sum / (len(dfs) + 1)
                forecast[i] = int_total[i]
                continue
            # forecast formula:
            # y_x+m = smooth + m*trend + season_indices_x-L+1+(m-1)modL
            if i >= tot_len:
                m = i - tot_len + 1
                s = 0
                for t in dfs:
                    s += HoltWinters.trend(t, season_len)
                mtrend = s / (len(dfs) + 1)
                # account for yearly population growth
                forecast[i] = (smooth + (m*mtrend) + season_indices[mod_L])
            else:
                pt = int_total[i]
                prev_smooth = smooth
                smooth = a * (pt - season_indices[mod_L]) 
                + (1 - a) * (smooth + trend)
                trend = b * (smooth - prev_smooth) + (1 - b) * trend
                season_indices[mod_L] = g * (pt - smooth) + (1 - g) * season_indices[mod_L]
                forecast[i] = smooth + trend + season_indices[mod_L]
        return forecast
hw = HoltWinters
c = DataInitialization()
(t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
both = [t196_17, t196_18, t196_19]
df2 = pd.concat(both)
d = df2.to_numpy()
h = hw.triple_exp_smooth([t196_17, t196_18, t196_19], 21, .54, .02, .86, 91)
plt.plot(h, label = "Forecasted")
t = t196_19.to_numpy()
plt.plot(d, label = "Actual")
plt.title("196th and 44th Actual/Predicted Traffic 2017-2020")
plt.legend()
plt.show()

def fitting(actual, df1, df2, df3):
    alpha = np.arange(start=0, stop=1, step=.02)
    beta = np.arange(start=0, stop=1, step=.02)
    gamma = np.arange(start=0, stop=1, step=.02)
    r_best = -20000
    best = ()
    for a in alpha:
        for b in beta:
            for g in gamma:
                h = hw.triple_exp_smooth(df1, df2, df3, 21, a, b, g, 0)
                sq_err = (d - h)**2
                sse = np.sum(sq_err)

                avg_act = np.mean(d)
                sqe = (d - avg_act) ** 2
                sst = np.sum(sqe)

                r_2 = 1 - sse/sst
                if r_2 > r_best:
                    best = (a, b, g)
                    r_best = r_2
                    print(str(r_best))
    return best

#print(fitting(d, t196_17, t196_18, t196_19))

