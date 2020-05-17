#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Data Initialization class
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
        t196_19, t200_19, t196_18 = None, None, None
        t200_18, t196_17, t200_17 = None, None, None
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
        else:
            return "Invalid plot type given"

    def busy_days(self, df):
        """ 
        Determines the days of the week with most and least traffic

        Parameters:
        df: DataFrame containing data about an intersection

        Returns:
        best_day: weekday with most traffic on average
        worst_day: day with least traffic on average (weekends included)
        worst_weekday: day with least traffic, excluding weekends
        """
        # find the day of week with highest traffic for each intersection
        df.index = pd.to_datetime(df.index, utc=True)
        count = df.groupby(df.index.dayofweek).sum()
        best_day = count.idxmax()
        days = ['M', 'T', 'W', 'TH', 'F', 'SAT', 'SUN']
        best_day = days[best_day[0]]
        
        # find the day of week with lowest traffic for each intersection
        worst_day = count.idxmin()
        worst_day = days[worst_day[0]]

        return (best_day, worst_day)

    def tests(self):
        """
        Tests data frame accuracy in a variety of ways
        """     
        # Test 1: confirm 91 days in each dataset after totalling
        print("Beginning testing for Data Initialization Class")
        (t196_19, t196_18, t196_17, 
         t200_19, t200_18, t200_17) = self.read_files()
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
