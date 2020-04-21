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
# Version: 1.3 - 4/21/2020
# Written for Python 3.7.2

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

class DataInitialization:
    def __init__(self):
        pass

    def main(self):
        # import csv files
        t196_19, t200_19, t196_18, t200_18 = None, None, None, None
        csv = ["196th_2019.csv", "200th_2019.csv", "196th_2018.csv", "200th_2018.csv"]
        intersections = [t196_19, t200_19, t196_18, t200_18]
        i = 0

        # store csv files in respective intersection object
        # format: datetime, int
        for file in csv:
            df = pd.read_csv(file, delimiter = ",", 
                             usecols = ["Time", "Int Total"])
            tdf = pd.DataFrame(data = df)
            tdf.drop(tdf.tail(4).index, inplace = True)
            tdf["Time"] = pd.to_datetime(tdf["Time"], utc=True)
            tdf.set_index("Time", inplace=True)
            intersections[i] = tdf
            i+=1

        (t196_19, t200_19, t196_18, t200_18) = intersections
        #n = pd.date_range(start='9/2/2019', end='11/30/2019')
        n1 = self.day_sum(t196_19)
        n2 = self.day_sum(t196_18)
        self.visualize(n1, "Stacked bar", n2)
       
    def visualize(self, df, ptype, df1):
        """
        Provides a variety of data visualizations
        Parameters:
        da: numpy array of daily totals
        dr: Datetime ndarray storing dates to be plotted
        ptype: String with the name of the plot type
        """
        z = pd.date_range(start="9/01/0", end="11/30/0")
        zlen = np.arange(len(z))
        compareType = ptype.lower()
        if compareType == "Plot".lower():
            # create a line plot using matplotlib
            fig, ax = plt.subplots()
            ax.plot(df, 'c')

            ax.set(xlabel="Date", ylabel='Total Daily Cars',
               title='Total Daily Cars at 196th and 44th Sept-Nov 2019'
               )
            ax.grid()
            plt.show()
        elif compareType == "Stacked Bar".lower():
            # creates a histogram using pandas

            # We can set the number of bins with the `bins` kwarg
            #plt.figure()
            fig, ax = plt.subplots()
            rects1 = ax.bar(z, df["Int Total"], label='2019')
            rects2 = ax.bar(z, df1["Int Total"], label='2018')
            ax.set_xticks(zlen)
            ax.set_xticklabels(z)
            plt.show()
        else:
            print("Invalid plot type given")


    def day_sum(self, df):
        """
        Calculates the daily totals using the hourly totals
        Parameters:
        df: DataFrame
        dates: Datetime ndarray storing the dates to be analyzed

        daily: Numpy array of daily totals
        """
        daily = df.groupby(df.index.date).sum()
        return daily
    
    def tests(df):
        """
        Tests data frame accuracy in a variety of ways
        """
      

    #def plots():
        #Individual plots:
        # Each intersection, each year
        # Plot on same graph:
        # 196th and 200th for each year
        # All yearly data on a shared graph
        # Histograms:
        # All yearly data on a shared graph
        #drange19 = pd.date_range(start='9/2/2019', end='11/30/2019')
        #drange18 = pd.date_range(start='9/2/2018', end='11/30/2018')
        #day196_19 = day_sum(t196_19, drange19)
        #day196_19 = day196_19[1:-1]
        #day200_19 = day_sum(t200_19, drange19)
        #day200_19 = day200_19[1:-1]
        #day196_18 = day_sum(t196_18, drange18)
        #day200_18 = day_sum(t196_18, drange18)
        #print(day196_19)
        #visualize(day196_19, drange19)

c = DataInitialization()
c.main()
