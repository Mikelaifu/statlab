## created by Mike WU
##  last Modified: 2/14/2019
## class to support statistic_lab_toolkits and statistic_lab_vizard 

import warnings
from functools import partial
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
