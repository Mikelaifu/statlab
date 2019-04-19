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
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
from collections import Counter
import math
from math import ceil, floor
from decimal import *

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
            return sls.float_round(StndrdDvtn, rnd, round) 

        elif Method == "computational":
            Svalues = list(map(lambda v: v ** 2, values))
            DeviSum = sum(Svalues)
            ObservSum = sum(values)
            StndrdDvtn = ((DeviSum - ((ObservSum ** 2) /Count) ) /Count) ** (1/2)
            return sls.float_round(StndrdDvtn, rnd, round)

    # ----------------------------------------------- Variance -------------------------------------------------
    @staticmethod
    def Variance(x, rm_na = False , type = "Sample",  convert_na = False, rnd = 2):
        std = statistic_lab_toolkits.std_dev(x = x, rm_na = rm_na, type = type, convert_na = convert_na , rnd = rnd)
        return sls.float_round(std * std, rnd, round)
    # ----------------------------------------------- Z-score -----------------------------------------------------------
    @staticmethod
    def z_score(mean, std, x, rnd = 3):
        z = (x - mean)/std
        return sls.float_round(z, 3, round)
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
        Max = sls.float_round(max(x), rnd, round)
        Min = sls.float_round(min(x), rnd, round)
        result = [Min, sls.float_round(Q1, rnd, round), sls.float_round(Q2, rnd, round), sls.float_round(Q3, rnd, round), Max]
        if Output == "IQR":
            return rsls.float_round(Q3 - Q1, rnd, round)
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
            return sls.float_round(lcc, rnd, round)

    # ----------------------------------------------- Linear Equation, formula & Plot -------------------------------------------------
    @staticmethod
    def LinearEquation(x_points, y_points, rnd = 4, x_range = (-5, 5), plot = False):
        m = (y_points[1]-y_points[0])/(x_points[1] - x_points[0])
        b = -(m * x_points[1])+ y_points[1]
        m1 = sls.float_round(m, rnd, round)
        b1 = sls.float_round(b, rnd, round)
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
            b1= sls.float_round((Sy/Sx) * r, rnd, round)
            b0 = Meany - (Meanx * b1)
            if plot == False and residual2 == False:
                return [rsls.float_round(b1, rnd, round), sls.float_round(b0, rnd, round)]
            elif plot == False and residual2 == True and y != None:
                predicted_y = list(map(lambda c: (c * b1) + b0, x))
                rsdl = list(map(lambda v, b : (v - b) ** 2, y, predicted_y))
                rsdl_sum = sum(rsdl)
                return [sls.float_round(b1, rnd, round), sls.float_round(b0, rnd,round), sls.float_round(rsdl_sum, rnd, round)]
            elif plot == True and x_range != None:
                title = "Least-Sqaured Regression: y = {} * x + {}".format(sls.float_round(b1, rnd, round), sls.float_round(b0, rnd, round))
                func = partial(sls.linear_equa_formula, b1, b0)
                return slv.Linear_graph(formula=func, x_range = x_range, title = title)
            elif plot == True and x_range == None:
                return warnings.warn("Warning, Missing Values in argument x_range")
        elif r == 0:
            return warnings.warn("Warning, r = 0, no linear relationship defined")
        else:
            return warnings.warn("Warning, r < -1 or 1 > 1, so, Could not define any relationship in applied data sets")


    # ----------------------------------------------- Counting Methods -------------------------------------------------
    # type = "multiplication", "permutation", "combination", "permutation" Nondisctinct, "permutation" disctinct
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

    # ----------------------------------------------- Discrete Mean/Discrete standerd Deviaiton -------------------------------------------------
    # x is the frequency list, and p is the probability value list corresponded to each frequency
    @staticmethod
    def discrete(x, p, type = "mean", rnd = 6):
        result_mean =  round(sum(list(map(lambda a, b: a * b, x, p))), rnd)
        if type == "mean":
            return result_mean
        if type ==  "std":
            result_std = sum(list(map(lambda a,b : ((a - result_mean) ** 2) * b , x, p)))
            final_result_std = result_std ** (1/2)
            return sls.float_round(final_result_std, rnd, round)
    # ----------------------------------------------- Binomial Probability Formula -------------------------------------------------
    @staticmethod
    def BinomP(n, x, p, rnd = 4):
        Count = statistic_lab_toolkits.Counting(n = n, r = x , type = "combination", repeated = False, ordered = False, distinct = True)
        BinomPro = Count * ((p) ** x) * ((1-p) ** (n -x))
        return sls.float_round(BinomPro, rnd, round)
    # ----------------------------------------------- Binomial Distrubution -------------------------------------------------
    @staticmethod
    #calculate  "probability", "mean" and "std"
    # x = [value1, value2] && compare = "="
    # compare <=, <, =, >, >=
    # if between two value, the 3rd element is deciding betwher inclusive
    def Binom(n,  p, x = None,compare = None, type= None, rnd = 4):
        if compare != None and type == "probability" and x != None:
            if compare == "=" and type == "probability":
                lst = sls.Opera_range_list(compare = compare, x = x, n = n)
            elif isinstance(x, (list,)) and compare in ["=", "!="]:
                lst = sls.Opera_range_list(compare = compare, x = x, n = n)
            else:
                lst = sls.Opera_range_list(compare = compare, x = x, n= n)
            probLst = []
            for ele in lst:
                BiP = statistic_lab_toolkits.BinomP(n = n, x = ele, p = p, rnd = rnd)
                probLst.append(BiP)
            result = functools.reduce(lambda x,y: x + y, probLst)
            return result

        if compare == None and type == None and x != None:
            result = statistic_lab_toolkits.BinomP(n = n, x = x, p = p, rnd = rnd)
            return result
        if type != "probability" and type != None :
            mn = n * p
            mean = rsls.float_round(mn, rnd, round)
            if type == "mean":
                return mean
            if type == "std":
                variance = mean * (1 - p)
                std_dev = variance**(1/2)
                return sls.float_round(std_dev, rnd, round)
    
    # ----------------------------------------------- Poisson Probability formula  -------------------------------------------------
    # lambda represnt the avreage number of occurrences of the event in some interval of length 1 and e=2.71828
    #T is length of time

    @staticmethod    
    def PoissonP(lmda, x, t, rnd = 4, rnde = 5):
        e = round(sls.e, rnde)
        total_occur = lmda * t
        FactoX = sls.Factorial_n(x)
        result = ((total_occur ** x)/FactoX) * (e ** (-total_occur))
        return sls.float_round(result, rnd, round)

    #----------------------------------------------- Poisson Probability distribution  -------------------------------------------------
    #type = calculate  "probability", "mean" and "std"
    @staticmethod   
    def Poisson(lmda, t, x= None, rnd = 4, rnde = 5, compare = None, type = None):
        if compare != None and type == "probability" and x != None:   

            if compare == "=" and type == "probability":
                lst = sls.Opera_range_list(compare = compare, x = x)
            elif isinstance(x, (list,)) and compare in ["=", "!="]:
                lst = sls.Opera_range_list(compare = compare, x = x)
            else:
                lst = sls.Opera_range_list(compare = compare, x = x)

            probLst = []
            for ele in lst:
                PisnP = statistic_lab_toolkits.PoissonP(lmda = lmda, x = ele, t = t, rnd = rnd, rnde = rnde)
                probLst.append(PisnP)
            result = functools.reduce(lambda x,y: x + y, probLst)
            return result

        if compare == None and type == None and x != None:
            result = statistic_lab_toolkits.PoissonP(lmda = lmda, x = ele, t = t, rnd = rnd, rnde = rnde)
            return result

        if type != "probability" and type != None and x == None:
            avg = lmda * t
            mean = sls.float_round(avg, rnd, round)
            if type == "mean":
                return mean
            if type == "std":
                std_dev = mean**(1/2)
                return sls.float_round(std_dev, rnd, round)
    #----------------------------------------------- Normal Distribution : probablity density function -------------------------------------------------
    # pdf function : (1/(stedev * (2*pi) ** (1/2)) * (e ** ((-1/2) * ((x - mean)/stdev)**2)))
    @staticmethod   
    def normpdf(z):
        constant = 1.0 / np.sqrt(2*math.pi)
        y = constant * np.exp((-z**2) / 2.0) 
        return y
    #----------------------------------------------- Normal Distribution : create z-table -------------------------------------------------
    @staticmethod   
    def norm_z_table(max_lim = 3.5):
        standerd_normal_table = pd.DataFrame(data = [], 
                                            index = np.round(np.arange(-3.5, max_lim + 0.1, .1), 2),
                                            columns = np.round(np.arange(0.00, .1, .01), 2))
        for index in standerd_normal_table.index:
            for column in standerd_normal_table.columns:
                z = np.round(index + column, 2)
                value, _ = quad(statistic_lab_toolkits.normpdf, np.NINF, z) 
                standerd_normal_table.loc[index, column] = value
        standerd_normal_table.index = standerd_normal_table.index.astype(str)
        standerd_normal_table.columns = standerd_normal_table.columns.astype(str)
        return standerd_normal_table


    
    #----------------------------------------------- Normal Distribution : nomral curve area claculation -------------------------------------------------
    # find normal distribution probability based on z-score  and pdf function to get proportion of area of distribution less than x
    # accoridng to inver z-score to find area_p
    # accoridng to area_p to find matched z-score
    # accoridng z  or range, find area_p
    # accorudng to calculation, to plot normal distribution and its targeted probability area 
    # when plot == true, we can only plot graph when we calculate probability area 

    @staticmethod   
    def normcurv (x = None, mean = None, std = None, z = None, area_p = None, InvNorm = False, rnd = 4, p = True, ztable = False, find_z = False, 
                     find_x = False, plot = False, compare = None):
        zList = None
        graph = None
        if isinstance(x, (list,)) and len(x) == 2 and z == None and  ztable == False and find_x == False:
            z1 = sls.float_round(statistic_lab_toolkits.z_score(mean, std, min(x)), 2, round)
            z2 = sls.float_round(statistic_lab_toolkits.z_score(mean, std, max(x)), 2, round)
            zList = [z1, z2]
        elif isinstance(x, (list,)) == False and z == None and ztable == False:
            z = sls.float_round(statistic_lab_toolkits.z_score(mean, std, x), 2, round)
            #print("1. z eqauls: {}".format(z))
        elif z != None and x == None and mean == None and std == None and ztable == False:
            if isinstance(z, (list,)) == False:
                z = z
            else:
                zList = z
        if ztable == True and find_x == False:
            table = statistic_lab_toolkits.norm_z_table(max_lim = 3.5)
            if find_z == True and area_p != None:
                if InvNorm == False:
                    area_p
                else:
                    area_p = 1- area_p
                area_p = sls.float_round(area_p, rnd, round)
                # print(area_p)
                result_list = sls.data_match(value = area_p, table = table, index=False)
                try:
                    result = sls.float_round(float(result_list[0]) + float(result_list[1]), 4, round)
                    return result 
                except:
                    print ("Error: could not find probability value in the Z-table !!")
                
            else:
                return table
                
        if p == True and (z != None or (x != None and mean != None and std != None)) and ztable == False and find_x == False:
            if isinstance(zList, (list,)) and len(zList) == 2:
                P1, _= quad(statistic_lab_toolkits.normpdf, np.NINF, zList[0])
                P2, _= quad(statistic_lab_toolkits.normpdf, np.NINF, zList[1])
                result = sls.float_round(P2 - P1, rnd, round)
                if plot == True and compare!= None :
                    graph = slv.norm_plot(z = zList, p=result,range = compare,  wdh =9 , hgt = 6, clr = "b")    
            else:
                #print("z eqauls: {}".format(z))
                P, _= quad(statistic_lab_toolkits.normpdf, np.NINF, z)
                result = sls.float_round(P, rnd, round)
                # print("result: {}".format(result))
                if plot == True and compare!= None :
                    if compare in [">", ">="]:
                        graph = slv.norm_plot(z = z, p= (1-result),range = compare,  wdh =9 , hgt = 6, clr = "b")  
                    else:
                        graph = slv.norm_plot(z = z, p= result,range = compare,  wdh =9 , hgt = 6, clr = "b")  
            if graph == None:
                #  print("result: {}".format(result))
                 return result 
            else:
                return graph 
        if find_x == True and z != None and mean != None and std != None and x == None:
                if isinstance(z, (list,)) == False:
                    result = (z * std) + mean  
                else:
                    val1 = (z[0] * std) + mean
                    val2 = (z[1] * std) + mean
                    result = [val1, val2]
                return result
    #----------------------------------------------- Gaussian Crtical Values or Critical value for Z -------------------------------------------------
    # P is tail area for confidnece interval: 95% as 0.025 , 90% as 0.05, 99% as 0.005
    @staticmethod
    def z_critical_val(value , rnd = 2):
            result = scipy.stats.norm.ppf(value)
            return result
       

     #----------------------------------------------- Confidence Interval in normal distibution -------------------------------------------------
    #  <95% of all sample proportions> will <result in>< confidence interval estimates> that <contain the population proportion>, 
    #  whereas <5% of all sample proportions> will <result in> <confidence interval estimates> that <do not contain the population proportion>
    # N is sample size, n is proportion of sample with certain characterristics, 
    # P is tail area for confidnece interval: 95% as 0.025 , 90% as 0.05, 99% as 0.005
    @staticmethod   
    def z_interval(n, x, alpha, rnd = 3, rndZ = 2):
        p = x/n
        value = (1-alpha) + (alpha/2)
        critical_val = sls.float_round(statistic_lab_toolkits.z_critical_val(value), places = rndZ, direction = round)
        # print(critical_val)
        standerd_error = (p*(1-p)/ n) ** (1/2)
        # print(standerd_error)
        margin_error = critical_val * standerd_error
        # print(margin_error)
        lower = sls.float_round(p - margin_error, 3, round)
        higher = sls.float_round(p + margin_error, 3, round)
        return (lower, higher)
    
    #----------------------------------------------- t-distribution: t-value -level of confidence for estimates of population mean ---------------------------------------------
    # confidence level: 95%, then Aplah would be 1-95% = 0.05
    @staticmethod   
    def t_crtical_value(degree_of_freedom, alpha, rnd=3):
        alpha_half = alpha/2
        t_value = scipy.stats.t.ppf(1-alpha_half, degree_of_freedom)
        t_value = sls.float_round(t_value, places= rnd, direction = round)
        return t_value
    
    #----------------------------------------------- t-distribution: t-interval- level of confidence for estimates of population mean ---------------------------------------------
    # confidence level: 95%, then Aplah would be 1-95% = 0.05
    # degree of reedom is usually sample size - 1 as n -1
    @staticmethod   
    def t_interval( alpha, x = None, rnd=3, degree_of_freedom = None, mean = None, std=None):
        if x == None and mean != None and std != None and degree_of_freedom != None :
            
            if degree_of_freedom < 30:
                n = degree_of_freedom + 1
            else:
                n = degree_of_freedom
            t_value = statistic_lab_toolkits.t_crtical_value(degree_of_freedom, alpha, rnd=3)
            piece = std/(n ** (1/2))
        if x != None :
            n = len(x)
            if n < 30:
                degree_of_freedom = n -1
            else:
                degree_of_freedom = n
            t_value = statistic_lab_toolkits.t_crtical_value(degree_of_freedom, alpha, rnd=3)
            mean = statistic_lab_toolkits.mean(x)
            std = statistic_lab_toolkits.std_dev(x, type = "Sample", rnd = rnd)
            piece = std/(n ** (1/2))

        margin_of_error = float(t_value) * float(piece)
        lower_bound = sls.float_round(mean - margin_of_error, places= rnd, direction = round)
        high_bound = sls.float_round(mean + margin_of_error, places= rnd, direction = round)
        return (lower_bound, high_bound)

    
    #----------------------------------------------- Deterine sample size ---------------------------------------------
    # confidence level: 95%, then Aplah would be 1-95% = 0.05
    # type = popu (sample size to estimae the population proportion); type = mean, sample size to estimate population mean, type = std, samep size to estimate population standerd deviation
    # E is margin of error
    @staticmethod   
    def sample_size_inference(alpha, s= None, p = None, E = None , type = "popu"):
        if type == "popu" and E != None:
            z_value = statistic_lab_toolkits.z_critical_val(alpha)
            # print((z_value/E) ** 2)
            if p == None:
                result = 0.25 * ((z_value/E) ** 2)     
            if p != None:
                result = p * (1-p) * ((z_value/E) ** 2)
            n = sls.float_round(result, 0, round)
            return n
        
        if type == "mean" and s != None and E != None:
            z_value = statistic_lab_toolkits.z_critical_val(alpha)
            result = ((z_value * s) / E) ** (2)
            n = sls.float_round(result, 0, round)
            return n
    # --------------------------------------------- P-value generating ----------------------------------------
    @staticmethod  
    def p_values_z(z, side = 1, compare = "<"):
        if side == None and compare == None:
            result = sls.float_round(scipy.stats.norm.sf(z), 4, round)
            return result
        if side == 1:
            if compare == "<" :
                if z <=0:
                    z = abs(z)
                result = sls.float_round(scipy.stats.norm.sf(abs(z)), 4, round)#one-sided in left
                
            if compare == ">" :
                if z >= 0 :
                    z = -z
                result = sls.float_round(scipy.stats.norm.sf(abs(z)), 4, round)
            return result 

        if side == 2 :
            if z < 0:
                z = abs(z)
            if z > 0:
                z = -z
            if compare == "<>":
                result = sls.float_round(scipy.stats.norm.sf(z), 4, round)
                p_val = result  * 2
                return p_val
            # if compare == "><":
            #     result_left = sls.float_round(scipy.stats.norm.sf(z_left), 4, round)
            #     result_right = sls.float_round(scipy.stats.norm.sf(z_right), 4, round)
            #     middle_area = result_right - result_left
            #     return middle_area
        
    #----------------------------------------------- Hypothesis Testing for population proportion ---------------------------------------------
    @staticmethod   
    # type= classic (compare z-crtical value)
    # type = p_value,  compare p_value with level of significant which is alpha
    # type =confidence Interval
    # compare = <, > , <>, each matches left tail, right tail and two tails
    def hypo_test_p(p0, p, n, alpha, type = "classic", compare = "<",rnd =3):
        result0 = "Null Hypothesis Rejected"
        result1 = "Not sufficient to Reject Null Hypothesis"
        z0 = sls.float_round((p - p0) / (p0 * (1-p0)/n) ** (0.5), 2, round)
        print(z0)
        if type == "classic":
            if compare =="<":
                z = statistic_lab_toolkits.z_critical_val(alpha , rnd = 3)
                if z0 < z:
                    return {result0: [z0, z]} ## null hypothesis got rejected
                else:
                    return {result1: [z0, z]} ## not sufficient to reject null hypothesis
            if compare == ">":
                z = - statistic_lab_toolkits.z_critical_val(alpha , rnd = 3)
                if z0 > z:
                    return {result0: [z0, z]} ## null hypothesis got rejected
                else:
                    return {result1: [z0, z]} ## not sufficient to reject null hypothesis

            if compare == "<>":
                alpha = alpha /2
                z_left = statistic_lab_toolkits.z_critical_val(alpha , rnd = 3)
                z_right = -statistic_lab_toolkits.z_critical_val(alpha , rnd = 3)
                if z0 > z_right and z0 < z_left:
                    return {result0: [z0, z_left, z_right]} ## null hypothesis got rejected
                elif z0 < z_right and z0 > z_left:
                    return {result1: [z0, z_left, z_right]} ## not sufficient to reject null hypothesis

        # when using p_value to do hypothesis testing    
        if type == "p_value":
            if compare =="<":
                p_val = statistic_lab_toolkits.p_values_z(z0, side = 1, compare = compare)
                if alpha > p_val:
                    return {result0: [p_val, alpha]}
                else:
                    return {result1: [p_val, alpha]}
            if compare == ">":
                p_val = statistic_lab_toolkits.p_values_z(z0, side = 1, compare = compare)
                if alpha < p_val:
                    return {result0: [p_val, alpha]}
                else:
                    return {result1: [p_val, alpha]}
            if compare == "<>":
                p_val = statistic_lab_toolkits.p_values_z(z0, side = 2, compare = compare)
                if alpha > p_val:
                    return {result0: [p_val, alpha]}
                elif alpha < p_val:
                    return {result1: [p_val, alpha]}
                








        



        

            
      






    



