#=============================================================================
# General Documentation
# Lynnwood Traffic Prediction
#
# Light Rail class
#
# Additional Documentation:
# Author: Callie Bianco
# Version: 1.11 - 5/6/2020
# Written for Python 3.7.2
#==============================================================================

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random as rand

class LightRail:
    """
    This class will randomly generate light-rail ridership based on ridership
    estimates from Sound Transit. It will also predict how this ridership
    might affect forecasted vehicle traffic.
    """
    def __init__(self):
        pass

    def weekday(self):
        """
        Generates random weekday light-rail usage 
        """
        choices = np.arange(0, 24, 1)
        # 5% in the AM - 6 periods
        morn = np.array([.005, .006, .008, .008, .01, .013])
        # 40% morning peak - 4 periods
        morn_peak = np.array([.07, .1, .13, .1])
        # 10% mid-day - 4 periods
        mid_day = np.array([.03, .02, .02, .03])
        # 40% afternoon peak - 4 periods
        aft_peak = np.array([.1, .13, .1, .07])
        # 5% night
        night = np.array([.013, .01, .008, .008, .006, .005])
        trip_probs = np.concatenate((morn, morn_peak, mid_day, aft_peak, night))
        a = np.random.choice(choices, size=10000, p=trip_probs)
        times = np.sort(a)
        plt.hist(times, bins=75)
        plt.show()
