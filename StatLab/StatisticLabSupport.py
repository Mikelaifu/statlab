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

class statistic_lab_support:
     # Supportive Method Supports another methods or can be supportive by another support function
     # Main Method can be used as an independent methods

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
    @staticmethod
    def Factorial_n(n):
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
    
    