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
# Version: 1.10 - 5/5/2020
# Written for Python 3.7.2
#==============================================================================

# module imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random as rand
import DataInit as di
from TES import HoltWinters

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
    HoltWinters.forecast_2026(hw)
    
