import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


#problem 3
D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/DC_noise1.xlsx')
list2 = D['n'].tolist()
len1 = len(list2)
list1 = []
for i in range(len1):
    list1.append([(D['n'].tolist())[i], (D['y'].tolist())[i]])
#len1 = len(list1)

#initial guess
slope = 5
intercept = 0

#formula for d/d(slope) = d/dm of (y- (mx+c))^2 = 2*(-x)(y - (mx+c))
# and for d/d(intercept) = d/dc of (y- (mx+c))^2 = 2*(-1)(y- (mx+c))

tot_steps = 0
slope_sum=0
intercept_sum = 0

while(tot_steps<1000):
    for item in list1:
        temp1 = 2*(-1*item[0])*(item[1]- (slope*item[0]+intercept))
        slope_sum += temp1
        temp2 = 2 * (-1 ) * (item[1] - (slope * item[0] + intercept))
        intercept_sum += temp2
    learning_slope = 0.000000001
    learning_int = 0.0001
    slope  = slope - slope_sum* learning_slope
    intercept = intercept - intercept_sum*learning_int
    tot_steps +=1
    print(slope, intercept, tot_steps)
    slope_sum = 0
    intercept_sum = 0
