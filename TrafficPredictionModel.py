#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Driver code. A project predicting and simulating traffic at two main
# intersections in Lynnwood: 196th/44th and 200th/44th.
# Will predict and analyze traffic impacts before and after
# addition of Link Light-Rail in 2024
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.15 - 5/13/2020
# Written for Python 3.7.2
#==============================================================================

# module imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import datetime as dt
import random as rand
from DataInit import DataInitialization
from TES import HoltWinters
from LightRail import LightRail

def data_reqs():
    """
    Run all code for 'Data' requirements
    """
    # running the program so far
    c = di.DataInitialization()
    (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
    c.main("C")
    c.tests()

def TES_reqs():
    """
    Run all code for 'TES' requirements
    """
    hw = HoltWinters()
    c = DataInitialization()
    (t196_19, t196_18, t196_17, t200_19, t200_18, t200_17) = c.read_files()
    
    r_196s = [t196_17, t196_18, t196_19]
    r_200s = [t200_18, t200_19]


    actual = pd.concat(r_196s)
    actual = actual.to_numpy()
    # plot the actual data for comparison
    plt.plot(actual, label = "Actual (Until 2019)")

    # linear trend line
    (modeled, t_future) = hw.triple_exp_smooth(r_196s, 21, .5, .01, .92, 0)
    x = np.arange(1, len(modeled)+1)
    z = np.polyfit(x, modeled, deg=1)
    p = np.poly1d(z)
    pts = np.arange(1, (len(modeled)*2.7)+1)
    eq = p(pts)
    slope = eq[2] - eq[1]
    print(slope)

    (points, model) = hw.forecast_2024(slp=slope, plot=True)
    plt.plot(pts, p(pts), label = "Trendline")
    plt.legend()
    plt.show()
    

def light_rail_reqs():
    """
    Run all code for 'Light-Rail requirements'
    """
    lr = LightRail()
    hw = HoltWinters()
    (future, model) = hw.forecast_2024(slp=10.3)
    (affecting, low, high) = lr.get_riders()
    #fig, ax = plt.subplots()
    plt.plot(low, label="Low Estimate")
    plt.plot(affecting, label= "Average Estimate")
    plt.plot(high, label= "High Estimate")
    plt.legend()
    plt.xlabel("Varying Bus Probabilities")
    tix = ['.05', '.10', '.15', '.20', '.25', '.30', '.35', '.40', '.45']
    tx = np.arange(0, 70, 7)
    plt.xticks(tx, tix)
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.ylabel("Number of Riders Using 196th and 44th")
    plt.title("How Changing Bus Probability Impacts Predicted \n Additional " +
              "Vehicle Traffic From Light-Rail Riders")
    plt.show()

def revenue_reqs():
    """
    Run all code for 'Revenue' requirements
    """
    lr = LightRail()
    (avg, low, high) = lr.avg_day()

    lr.revenue(avg, low, high)

#light_rail_reqs()
#TES_reqs()
revenue_reqs()


