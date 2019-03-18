## created by Mike WU
##  last Modified: 2/14/2019
## class to support visualizing Statistic Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import partial
from collections import Counter
import scipy
from scipy.integrate import quad
from scipy.stats import norm


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
    # @staticmethod
    # def bard_plot():
    #     pass


    # ----------------------------------------------- Frequency Histogram chart ------------------------------------------

    # ----------------------------------------------- normal distribution Chart ------------------------------------------
    @staticmethod
    def norm_plot(z, p=None,range= None, style = "fivethirtyeight", wdh =9 , hgt = 6, clr = "b"):

        if range == "<>" and isinstance(z, (list,)):
            z1 = round(z[0], 2)
            z2 = round(z[1], 2)
            z_text = "{} < z < {}".format(z1, z2)  
        if range in ["=", '<=', "<"] and isinstance(z, (list,)) == False:
            z1 = -3.5
            z2 = round(z, 2)
            z_text = "{} <= z".format(z2)
        if range in [">=", ">"] and isinstance(z, (list,)) == False:
            z1 = round(z, 2)
            z2 = 3.5
            z_text = "z <= {}".format(z1)
        x_plot = np.arange(z1, z2, 0.001)  
        x_all = np.arange(-5, 5, 0.001) 
        # mean = 0, stddev = 1, since Z-transform was calculated
        y = norm.pdf(x_plot, 0, 1)
        y2 = norm.pdf(x_all, 0, 1)

        fig, ax = plt.subplots(figsize = (wdh,hgt))
        if style == None:
            pass
        else:
            plt.style.use(style)
        ax.plot(x_all, y2)
        ax.fill_between(x_plot, y, 0, alpha = 0.3, color = clr)
        ax.fill_between(x_all, y2, 0, alpha = 0.1)
        ax.set_xlim([-5,5])
        ax.set_xlabel('# of Standard Deviations Outside the Mean')
        ax.set_yticklabels([])
        ax.set_title('Normal Curve')
        if p != None:
            text = 'probility = {}\n {}'.format(p, z_text)
        if p == None:
            text = '{}'.format(z_text)
        plt.text(-1.1, .15, text)
        plt.show()


    


    # ----------------------------------------------- Time Series line chart ------------------------------------------




