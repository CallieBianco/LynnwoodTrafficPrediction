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
# Version: 1.4 - 4/22/2020
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
        (t196_19, t196_18, t200_19, t200_18) = self.read_files()
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

    def read_files(self):
        # import csv files
        t196_19, t200_19, t196_18, t200_18 = None, None, None, None
        csv = ["196th_2019.csv", "200th_2019.csv", 
               "196th_2018.csv", "200th_2018.csv"]
        intersections = [t196_19, t200_19, t196_18, t200_18]
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

        (t196_19, t200_19, t196_18, t200_18) = intersections
        # obtain daily counts instead of hourly
        t196_19 = self.day_sum(t196_19)
        t196_18 = self.day_sum(t196_18)
        t200_19 = self.day_sum(t200_19)
        t200_18 = self.day_sum(t200_18)
        return (t196_19, t196_18, t200_19, t200_18)
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
        z = pd.date_range(start="9/01/0", end="11/30/0")
        zlen = np.arange(len(z))
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
c = DataInitialization()
#c.main(visual="B")

# tests
#c.tests()
