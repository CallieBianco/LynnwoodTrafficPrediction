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
# Version: 1.20 - 5/26/2020
# Written for Python 3.7.2
#==============================================================================

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random as rand
from DataInit import DataInitialization
from TES import HoltWinters
from LightRail import LightRail

def data_reqs():
    """
    Run code for 'Data' requirements
    """
    di = DataInitialization()

    # graph of choice
    #   A: individual plots of the intersection
    #   B: plots with years stacked on top
    #   C: plots the 2017 years
    #   D: all years on one graph
    #   None: returns nothing
    di.main(type="D")

    # runs tests
    di.tests()

def TES_reqs():
    """
    Run code for 'TES' requirements
    """
    hw = HoltWinters()

    # different functions:
    # A: plots the 2024 forecasted data w/actual data
    #    and actual data trendline
    # B: plots the seasonal averages to 2024
    # C: finds best values of alpha, beta, and gamma (takes awhile)
    hw.main(type="A")

def light_rail_reqs():
    """
    Run code for 'Light-Rail requirements'
    """
    lr = LightRail()
    # different functions
    # A: plots the histogram of expected avg daily ridership
    # B: graphs sensitivity analysis for bus probability variable
    # C: plots the impact on vehicle traffic 2024 from light-rail
    lr.main(type="C")

def revenue_reqs():
    """
    Run code for 'Revenue' requirements
    """
    lr = LightRail()
    (avg, low, high) = lr.avg_day()

    lr.revenue(avg, low, high)


#data_reqs()
#TES_reqs()
#light_rail_reqs()
#revenue_reqs()


