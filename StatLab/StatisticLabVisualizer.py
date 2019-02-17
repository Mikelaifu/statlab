## created by Mike WU
##  last Modified: 2/14/2019
## class to support visualizing Statistic Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import partial
from collections import Counter
class statistic_lab_vizard():
    # Supportive Method Supports another methods or can be supportive by another support function
    # Main Method can be used as an independent methods

    ## trouble shot to make sure Y axis start from 0

    def __init__(self):
        pass 
    
    # ----------------------------------------------- Linear Graph (Supportive) ------------------------------------------
    @staticmethod
    def Linear_graph(formula, x_range, title):
        x = np.array(x_range)
        y = formula(x)
        plt.plot(x, y)
        plt.title(title)
        plt.show()
    # ----------------------------------------------- Boxplot (Main & Supportive) ------------------------------------------
    @staticmethod
    def boxplot( title = None, labels = None, **kwargs):
        lst = []
        for key, value in kwargs.items():
            if key == None or value == None or value ==[]:
                return warnings.warn("Warn: Please input legit data for plotting")
            if isinstance(value, list) == False:
                value = list(value)
            if key != None and isinstance(value, list) == True and value != []:
                lst.append(value)

        if labels != None:
            plt.boxplot(lst, labels = labels )
        else:
            plt.boxplot(lst)
        if title != None:
            plt.title(title)
        plt.show()

    # ----------------------------------------------- Reguler Bar Chart ------------------------------------------


    # ----------------------------------------------- Frequency Histogram chart ------------------------------------------


    # ----------------------------------------------- Time Series line chart ------------------------------------------




