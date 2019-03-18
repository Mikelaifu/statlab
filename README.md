# statlab Python Module

## Documentation/Instructions of statlab Module:

#### In side of statlab module, there are functions we can use to perform statistical analysis such as mean, median, mode, find_IQR, LinearEquation...

mean(x, rm_na = False, convert_na = False)

median(x, rm_na = False)

mode(x, rm_na = False)
       
range(x, rm_na = False)
       
std_dev(x, Method = "conceptual", type = "Sample", rm_na = False,  convert_na = False, rnd = 4)
   
Variance(x, rm_na = False , type = "Sample",  convert_na = False, rnd = 2)
   
z_score(mean, std, x, rnd = 3)

emperical_rule (x, m, std, shape = "bell", K = None)
   
find_IQR(x, rm_na = False, rnd = 5, convert_na = False, Output = "IQR")
   
find_outliner(x, rm_na = False,  convert_na = False, plot = False, title = None, labels = None)
    
LCC(x, y, rnd = 4):

LinearEquation(x_points, y_points, rnd = 4, x_range = (-5, 5), plot = False)

LS_regression(x = None, y = None, S_x = None, S_y = None, Mean_x = None, Mean_y = None, R_LCC =None, rnd = 4, x_range = None, plot = False, residual2 = False)

Counting (n, r = None, k = None, type = "multiplication", repeated = False, ordered = False, distinct = False)

discrete(x, p, type = "mean", rnd = 6)

BinomP(n, x, p, rnd = 4)

Binom(n,  p, x = None,compare = None, type= None, rnd = 4) 

PoissonP(lmda, x, t, rnd = 4, rnde = 5)

Poisson(lmda, t, x= None, rnd = 4, rnde = 5, compare = None, type = None)

normpdf(z)
       
norm_z_table(max_lim )

normcurv (x = None, mean = None, std = None, z = None, area_p = None, InvNorm = False, rnd = 4, p = True, ztable = False, find_z = False, 
                     find_x = False, plot = False, compare = None)
        






    




