from StatisticLabSupport import statistic_lab_support
from StatisticLab import statistic_lab_toolkits
from StatisticLabVisualizer import statistic_lab_vizard
a = [7,6,1,7,9]
b = [3,2,6,9,5]  
x=[-2,-1,0,1,2]
y=[-2,1,4,5,5]
z = [100, 120, 400, 100000, -1, -200000, 400, 4700]
Flight = [7.43,7.21, 8.69, 8.64, 9.76, 6.85,
          7.89, 9.3, 8.03, 7, 8.8, 6.39, 7.54]
Control = [8.65, 6.99, 8.4, 9.66, 7.62, 7.44,
           8.55,9.88, 9.94, 7.14, 9.14]
x1 = [-1,1]
y1 = [1,5]
test1 = statistic_lab_toolkits()
test_viz = statistic_lab_vizard()
#print("boxplot: {}".format(test_viz.boxplot(ctl= Control, flgt = Flight, title = "Control vs Flight", labels = ["Control", "Flights"])))
#print("Outliner1: {}".format(test1.find_outliner(Flight,plot = False, title = None, labels = None)))

#print("Outliner2: {}".format(test1.find_outliner(Flight, plot = True, title = "Flights-boxplot", labels = ["flight"])))
print("LS_regression: {}".format(test1.LS_regression(x, y, rnd = 5, x_range = None, plot = False, residual2 = True)))
#print("LS_regression2: {}".format(test1.LS_regression( S_x = 2.5033, S_y = 1.8195, Mean_x = 3.3333, Mean_y = 4.1667, R_LCC = -0.9499, x_range = (0,6), plot = False)))
#print("Linear Eqaution: {}".format(test1.LinearEquation(x_points = x1, y_points = y1,  x_range = (-1,2), plot = True )))
#print("LS_regression: {}".format(test1.LS_regression(x, y, rnd = 4, x_range = (99,110), plot = True)))
# print("Mean: {}".format(test1.mean(b)))
# print("Mode: {}".format(test1.mode(b)))
# print("Median: {}".format(test1.median(b)))
# print("Standerd Deviation: {}".format(test1.std_dev(b)))
# print("Standerd Deviation2: {}".format(test1.std_dev(a, rm_na = True)))
# print("Standerd Deviation3: {}".format(test1.std_dev(a, rm_na = False, convert_na=True)))
# print("Variance: {}".format(test1.Variance(a, rm_na = False, convert_na=True)))
# print("Z-score: {}".format(test1.z_score(187.56, 0.396, 186.62)))
#print("LCC: {}".format(test1.LCC(x, y, rnd = 4)))
# print(test1.LinearEquation(x_points = x1, y_points = y1, rnd = 2, x_range = (-10,11), plot = False ))
# print("IQR: {}".format(test1.find_IQR(y, rm_na = False, rnd = 2, convert_na = False, Output = "IQR")))
# print("IQR: {}".format(test1.find_IQR(y, rm_na = False, rnd = 2, convert_na = False, Output = "5_sum")))

