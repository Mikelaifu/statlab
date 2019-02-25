## created by Mike WU
##  last Modified: 2/14/2019
## class to return Statistic Values/graphs/concepts/calculation
from StatisticLabSupport import statistic_lab_support
from StatisticLabVisualizer import statistic_lab_vizard
import warnings
from functools import partial
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

## call StatisticLabSupport.py  &  StatisticLabVisualizer.py
sls = statistic_lab_support()
slv = statistic_lab_vizard()

class statistic_lab_toolkits():
    
    def __init__(self):
         pass

    # ----------------------------------------------- MEAN -------------------------------------------------
    @staticmethod
    def mean(x, rm_na = False, convert_na = False):
        x = list(x)
        if rm_na == False:
            if convert_na == False and None in x :
                return warnings.warn("Warning: Please Convert Null value to 0")
            elif convert_na == True:
                ## if there is null value in the list, that counted into calculation, we convert it into zero
                values = list(map(lambda v: sls.convertNull(v), x))
                return sum(values)/len(values)
            else:
                return sum(x)/len(x)
        elif rm_na == True:
            if None in x:
                x.remove(None)
            return sum(x)/len(x)

    # ----------------------------------------------- Median -------------------------------------------------
    @staticmethod  
    def median(x, rm_na = False):
        x = list(x)
        if rm_na == True:
            if None in x:
                x.remove(None)
        n = len(x)
        for i in range(n):
            for j in range(0, n-i-1):
                if x[j] > x[j +1]:
                    temp = x[j]
                    x[j] = x[j +1]
                    x[j + 1] = temp
        v = len(x)
        if (v & 1 == 0):
            MP = ((v)/2 + (v)/2 + 1)/2 -1
            M =(x[int(MP-0.5)] + x[int(MP+0.5)])/2
            return M
        elif(v & 1 == 1 ):
            M = (v + 1)/2 -1
            return x[int(M)-1]

    
    # ----------------------------------------------- Mode -------------------------------------------------
    @staticmethod  
    def mode(x, rm_na = False):
        x = list(x)
        if rm_na  == True:
            if None in x:
                x.remove(None)
        md_index = dict(Counter(x))
        maxCount = max(md_index.values())
        Keys = list(set(x))
        compare = maxCount
        Md = []
        for ky in Keys:
            count = md_index.get(ky)
            if count == compare:
                Md.append(ky)
            else:
                continue
        if len(Md) < len(Keys):
            return Md[0]
        else:
            return "Warning: No Mode"
    
    # ----------------------------------------------- Range -------------------------------------------------
    @staticmethod      
    def range(x, rm_na = False):
        x = list(x)
        if rm_na  == True:
            if None in x:
                x.remove(None)
        return max(x) - min(x)
    
    # ----------------------------------------------- Standerd Deviation -------------------------------------------------
    @staticmethod
    def std_dev(x, Method = "conceptual", type = "Sample", rm_na = False,  convert_na = False, rnd = 4):
        x = list(x)
        if rm_na == False and convert_na == True:
            values = list(map(lambda v: sls.convertNull(v), x))
            Avg = statistic_lab_toolkits.mean(values)
        elif rm_na == True:
            if None in x:
                values = x
                values.remove(None)
                Avg = statistic_lab_toolkits.mean(values)
        else:
            values = x
            Avg = statistic_lab_toolkits.mean(values)
        if type == "Population":
            Count = len(values)  
        elif type == "Sample":
            Count = len(values) - 1 

        Deviation = list(map(lambda v : v - Avg, values))
        if Method == "conceptual":

            ## Step 2  get all the Deviation about the mean and sqaure them
            SDeviation = list(map(lambda v: v ** 2, Deviation))
            ## STep 3 add all the Sqaured Deviation and deveided by count 
            StndrdDvtn = (sum(SDeviation)/Count) ** (1/2)
            return round(StndrdDvtn, rnd) 

        elif Method == "computational":
            Svalues = list(map(lambda v: v ** 2, values))
            DeviSum = sum(Svalues)
            ObservSum = sum(values)
            StndrdDvtn = ((DeviSum - ((ObservSum ** 2) /Count) ) /Count) ** (1/2)
            return round(StndrdDvtn, rnd)

    # ----------------------------------------------- Variance -------------------------------------------------
    @staticmethod
    def Variance(x, rm_na = False , type = "Sample",  convert_na = False, rnd = 2):
        std = statistic_lab_toolkits.std_dev(x = x, rm_na = rm_na, type = type, convert_na = convert_na , rnd = rnd)
        return round(std * std, rnd)
    # ----------------------------------------------- Z-score -----------------------------------------------------------
    @staticmethod
    def z_score(mean, std, x, rnd = 3):
        z = (x - mean)/std
        return round(z, 3)
    # ----------------------------------------------- Percentile Scope (empirical rule + chebyshev's Inequality) -------------------------------------------------
    @staticmethod
    def emperical_rule (x, m, std, shape = "bell", K = None):
        if shape != "bell":
            return warnings.warn("Warning: Empirical rule doesnt apply thos non-bell shaped distribution")
        elif shape == "bell":
            return {1: 0.68, 2: 0.95, 3: 0.997}
        elif shape == "ChebyShev" and K > 1:
            Pctg = (1 - 1/K ** 2)
            valow = mean - (K * std)
            valhigh = mean - (K * std)
            info =  "at least {} of data lie between {} and {}  for K > 1".format(Pctg, valow, valhigh)
            return info

    # ----------------------------------------------- InterQuartile & 5 Number Summery -------------------------------------------------
    @staticmethod
    def find_IQR(x, rm_na = False, rnd = 5, convert_na = False, Output = "IQR"):
        x = list(x)
        if rm_na  == True:
            if None in x:
                x.remove(None)
        if rm_na == False and convert_na == True:
            x = list(map(lambda v: sls.convertNull(v), x))
        Q2 = statistic_lab_toolkits.median(x)
        Q2_bottom = list(filter(lambda x : x <= Q2, x))
        Q2_top = list(filter(lambda x : x >= Q2, x))
        Q1 = statistic_lab_toolkits.median(Q2_bottom)
        Q3 = statistic_lab_toolkits.median(Q2_top)
        Max = round(max(x), rnd)
        Min = round(min(x), rnd)
        result = [Min, round(Q1, rnd), round(Q2, rnd), round(Q3, rnd), Max]
        if Output == "IQR":
            return round(Q3 - Q1, rnd)
        elif Output == "5_sum":
            return result

    # ----------------------------------------------- return Outliner -------------------------------------------------
    @staticmethod
    def find_outliner(x, rm_na = False,  convert_na = False, plot = False, title = None, labels = None):
        IQR = statistic_lab_toolkits.find_IQR(x, rm_na = rm_na, convert_na = convert_na, Output = "IQR")
        five_sum = statistic_lab_toolkits.find_IQR(x, rm_na = rm_na, convert_na = convert_na, Output = "5_sum")
        TP3 = five_sum[3] + (1.5 * IQR)
        TP1 = five_sum[1] - (1.5 * IQR)
        OutLiner_bottom = list(filter(lambda a : a < TP1, x))
        OutLiner_top = list(filter(lambda b : b > TP3, x))
        if plot == False:
            return {"Outliner_Bottom": OutLiner_bottom, "Outliner_Top": OutLiner_top}
        if plot == True and title == None and labels == None :
            return slv.boxplot(data = x, title = None, labels = None)
        elif plot == True and title != None and labels != None :
            return slv.boxplot(data = x, title = title, labels = labels)
 
    
    # ----------------------------------------------- Linear Correlation Coefficient ------------------------------------------------
    @staticmethod
    def LCC(x, y, rnd = 4):
        if len(x) != len(y):
            return warnings.warn("Warning: x and y is not same length")
        else:
            Mx = statistic_lab_toolkits.mean(x)
            My = statistic_lab_toolkits.mean(y)
            n = len(x)
            Sx = statistic_lab_toolkits.std_dev(x, type = "Sample")
            Sy = statistic_lab_toolkits.std_dev(y, type = "Sample")
            
            X_z_score = partial(statistic_lab_toolkits.z_score, Mx, Sx )
            Y_z_score = partial(statistic_lab_toolkits.z_score, My, Sy )
            Xz = list(map(lambda c : X_z_score(c), x))
            Yz = list(map(lambda d : Y_z_score(d), y))
            Z_sum = sum(list(map(lambda e,d : e*d, Xz, Yz)))
            lcc =Z_sum/(n-1)
            return round(lcc, rnd)

    # ----------------------------------------------- Linear Equation, formula & Plot -------------------------------------------------
    @staticmethod
    def LinearEquation(x_points, y_points, rnd = 4, x_range = (-5, 5), plot = False):
        m = (y_points[1]-y_points[0])/(x_points[1] - x_points[0])
        b = -(m * x_points[1])+ y_points[1]
        m1 = round(m, rnd)
        b1 = round(b, rnd)
        equation= "y = {}x + {}".format(m1,b1)
        if plot == False:
            return [m1, b1]
        else:
            if x_range == None:
                warnings.warn("Warning: Missing value for X_range")
            else:
                func = partial(sls.linear_equa_formula, m1, b1)
                title = "{} ".format(equation)
                return slv.Linear_graph(formula=func, x_range = x_range, title = title )

    # ----------------------------------------------- Least-Sqaure Regression Equation & Graph -------------------------------------------------
    @staticmethod
    def LS_regression(x = None, y = None, S_x = None, S_y = None, Mean_x = None, Mean_y = None, R_LCC =None, rnd = 4, x_range = None, plot = False, residual2 = False):
        if S_x == None and S_y == None and Mean_x == None and Mean_y == None and R_LCC == None and x != None and y != None:
            Sx = statistic_lab_toolkits.std_dev(x)
            Sy = statistic_lab_toolkits.std_dev(y)
            Meanx = statistic_lab_toolkits.mean(x)
            Meany = statistic_lab_toolkits.mean(y)
            r = statistic_lab_toolkits.LCC(x, y)
        elif S_x != None and S_y != None and  Mean_x != None and Mean_y != None and R_LCC != None :
            Sx = S_x
            Sy = S_y
            Meanx = Mean_x
            Meany = Mean_y
            r = R_LCC
        if r >= -1 and r <= 1 and r != 0:
            b1= round((Sy/Sx) * r, rnd)
            b0 = Meany - (Meanx * b1)
            if plot == False and residual2 == False:
                return [round(b1, rnd), round(b0, rnd)]
            elif plot == False and residual2 == True and y != None:
                predicted_y = list(map(lambda c: (c * b1) + b0, x))
                rsdl = list(map(lambda v, b : (v - b) ** 2, y, predicted_y))
                rsdl_sum = sum(rsdl)
                return [round(b1, rnd), round(b0, rnd), round(rsdl_sum, rnd)]
            elif plot == True and x_range != None:
                title = "Least-Sqaured Regression: y = {} * x + {}".format(round(b1, rnd), round(b0, rnd))
                func = partial(sls.linear_equa_formula, b1, b0)
                return slv.Linear_graph(formula=func, x_range = x_range, title = title)
            elif plot == True and x_range == None:
                return warnings.warn("Warning, Missing Values in argument x_range")
        elif r == 0:
            return warnings.warn("Warning, r = 0, no linear relationship defined")
        else:
            return warnings.warn("Warning, r < -1 or 1 > 1, so, Could not define any relationship in applied data sets")


    # ----------------------------------------------- Counting Methods -------------------------------------------------
    # type = "multiplication", "permutation", "combination"
    # k should be a list of indistinct objects
    # n can be a value or a list of value
    # r should be a value
    @staticmethod
    def Counting (n, r = None, k = None, type = "multiplication", repeated = False, ordered = False, distinct = False):
        if isinstance(n, (list,)):
            if 0 in n:
                return "n can not be 0"
        if r != None and isinstance(n, (list,)) == False:
            if r > n:
                return "r can not be bigger than n"
            if r >=0 and n >= 0:
                pass
        if isinstance(n, (list,)) and type == "multiplication" and repeated == False and r == None:
            result = functools.reduce(lambda x,y: x*y, n)
            return result
        if isinstance(n, (list,)) == False and type == "multiplication" and repeated == True and r != None:
            result = n ** r
            return result
        if isinstance(n, (list,)) == False and type == "multiplication" and repeated == False and r != None:
            Lst = list(range(1,n+1))[-r:]
            result = functools.reduce(lambda x,y: x*y, Lst)
            return result
        if isinstance(n, (list,)) == False:
            ## when permutation but objects are distinct
            if type == 'permutation' and repeated == False and r != None and ordered == True and distinct == True:
                result = sls.Factorial_n(n)/sls.Factorial_n(n-r)
                return result
            ## when permutation but objects are not distinct
            if type == 'permutation' and repeated == False and isinstance(k, (list,)) and ordered == False and distinct == False:
                uniq_k_len = len(list(set(k)))
                k_obj = list(sls.kind_num(k).values())
                kind_Factorial = list(map(lambda a : sls.Factorial_n(a), k_obj))
                denominator = functools.reduce(lambda c,d: c*d, kind_Factorial)
                result = sls.Factorial_n(n)/denominator 
                return result
            if type == 'combination' and repeated == False and r != None and ordered == False and distinct == True:
                result = sls.Factorial_n(n)/(sls.Factorial_n(r) * sls.Factorial_n(n-r))
                return result

        




