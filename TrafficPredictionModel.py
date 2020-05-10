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
# Version: 1.12 - 5/9/2020
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
    noisy = [1, 1.5, 2]
    for noise in noisy:
        HoltWinters.forecast_2026(hw, n=noise)
    v = [t196_17, t196_18, t196_19]
    d = pd.concat(v)
    d = d.to_numpy()
    plt.plot(d, label = "Actual (Until 2019)")

    (modeled, b) = hw.triple_exp_smooth(v, 91, .54, .02, .86, 0)
    x = np.arange(1, len(modeled)+1)
    z = np.polyfit(x, modeled, deg=1)
    p = np.poly1d(z)
    pts = np.arange(1, (len(modeled)*3)+1)
    plt.plot(pts, p(pts), label = "Trendline")
    plt.legend()
    plt.show()
    

def light_rail_reqs():
    """
    Run all code for 'Light-Rail requirements'
    """
    lr = LightRail()
    lr.weekday()

light_rail_reqs()