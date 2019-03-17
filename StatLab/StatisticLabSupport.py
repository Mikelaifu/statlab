## created by Mike WU
##  last Modified: 2/14/2019
## class to support statistic_lab_toolkits and statistic_lab_vizard 

import warnings
from functools import partial
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
import scipy
from scipy.integrate import quad

class statistic_lab_support:
     # Supportive Method Supports another methods or can be supportive by another support function
     # Main Method can be used as an independent methods

    # set variable e to reprenst e value (e value default to be approximately 2.71828)
    e = math.e ## == np.exp()
    pi = math.pi
    def __init__(self):
        pass 
    # ----------------------------------------------- convert Null value to 0 (Supportive) ------------------------------------------
    @staticmethod
    def convertNull(v):
            if v == None:
                v = 0
                return v
            else:
                return v
    # ----------------------------------------------- output a linear formula (Supportive) ------------------------------------------
    @staticmethod
    def linear_equa_formula(m,b,x):
        return (x * m) + b
    
    # ----------------------------------------------- Factorial calculation: N! -------------------------------------------------
    # this function mainly assist Counting method in Main Stat lab class
    @staticmethod
    def Factorial_n(n):
        if n >= 0:
            if n == 0:
                return 1
            else:
                N= list(range(1, n+1))
                v = functools.reduce(lambda x,y: x*y, N)
                return v
    # ----------------------------------------------- count numbers of unique value of elements from a list -------------------------------------------------
    # return a dictionary
    @staticmethod
    def kind_num(lst):
        kinds = list(set(lst))
        count = {}
        for kind in kinds:
            count[kind] = 0
            for i in lst:
                if kind == i:
                    count[kind]+= 1
        return count
    
    # ----------------------------------------------- return range list based on operation (<, =, <= ...) -------------------------
    # compare : =, !=, <=, >=, >, <
    # this function mainly assist Statlab 's distributionn probability function/method calculation
    @staticmethod
    def Opera_range_list(compare, x, n = None):
        if compare == ">=" and n != None :
            lst = list(range(x, n + 1))
        if compare == ">" and n != None:
            lst = list(range(x + 1, n + 1))
        if compare == "<=" :
            lst = list(range(0, x + 1))
        if compare == "<" :
            lst = list(range(0, x))
        if compare == "=" :
            lst = [x]
        if isinstance(x, (list,)) and len(x) == 2:
            if compare == "=":
                lst = list(range(min(x), max(x) + 1))
            if compare == "!=":
                lst = list(range((min(x) + 1), max(x)))
        return lst 
    
    