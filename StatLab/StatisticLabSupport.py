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
    # --------------------------- by matching values within a table, return specific values 's index/col from a table -------------------------
    # find specific value from a table and return the values's index and col value or position
    # wehn index = False, the function return actual row name, and column name
    # when index = True, the function return actual row 's index and col 's index
    @staticmethod
    def data_match(value, table, index=False):
        try :
            row_val = list(table.index)
            col_val = list(table.columns)
            for row, col in table.iterrows():
                row_idx = row_val.index(row)
                for i in range(0, len(col)):
                    if round(col[i], 4) == value: 
                        col_name = col_val[i]
                        col_idx = i
                        result = [(row, col_name), (row_idx, col_idx )]
                        if index == False:
                            return result[0]
                        if index == True:
                            return result[1]
        except:
            print("Error: could not match value's index position within the table")



    # --------------------------- by matching col, row's index or name, return specific values from a table -----------------------
    # when index == True, mtaching use row index and col index to return value
    # when index == False, matching row/col name to return value
    # when enter row col pistion, index + 1 = current position
    @staticmethod
    def index_match(row, col, table, colnm = False, rownm = False, index = True):
        try :
            if index == True:
                if colnm == False and rownm == False:
                    result = table.iloc[row -1].iloc[col -1] 
                elif colnm == False and rownm == True:
                    result = table.iloc[row -1][col] 
                elif colnm == True and rownm == False:
                    result = table.loc[row].iloc[col-1]     
            else:
                result = table.loc[row].loc[col]
            return result
        except:
            print("Error: there is no matching values based on the col/row position provided")



        
        

        
