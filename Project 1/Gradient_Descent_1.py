import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

'''
#PROBLEM-1

D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/DC_noise1.xlsx')
list1 = D['y'].tolist()


# the slope is given by -2*(y(n)-A)
#Let the initial guess of A be 0

sum =0
count = 0
A=0
step_size =1
tot_steps = 0

while(tot_steps<1000):
    for i in list1:
        temp = -2*(i-A)
        sum += temp
    learning_rate = 0.00001
    step_size = learning_rate * sum
    A = A - step_size
    temp =0
    sum=0
    print(A, "@ step",tot_steps)
    tot_steps +=1

print("The value of A is ", A)


'''

'''

# 2nd Problem
D = pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/DC_noise2.xlsx')
list2 = D['y'].tolist()
list2.sort()
sum2 = 0
count2 = 0

for i in list2:
    count2 += 1

Q1 = (list2[int(count2 / 4)])
Q3 = (list2[int(3 * count2 / 4)])

iqr = Q3 - Q1

val1 = Q1 - 1.5 * iqr
val2 = Q3 + 1.5 * iqr

list4 = []

for i1 in list2:
    if ((i1 > val1) and (i1 < val2)):
        list4.append(i1)

len1 = len(list4)

sum = 0
count = 0


# the slope is given by -2*(y(n)-A)
#Let the initial guess of A be 0

A=0
step_size =1
tot_steps = 0
print("PROCESS 1")
while(tot_steps<1000):
    for i in list4:
        temp = -2*(i-A)
        sum += temp
    learning_rate = 0.001
    step_size = learning_rate * sum
    A = A - step_size
    temp =0
    sum=0
    print(A, "@ step",tot_steps)
    tot_steps +=1

print("The value of A is ", A)

list2 = np.array(list2)
mean = np.mean(list2)
std = np.std(list2)


outliers =[]
non_outliers =[]
for item in list2:
    z_score = (item -mean) /std
    if (np.abs(z_score)>3):
        outliers.append(item)
    else:
        non_outliers.append(item)


sum = 0
count = 0
# the slope is given by -2*(y(n)-A)
#Let the initial guess of A be 0
A=0
step_size =1
tot_steps = 0
print("PROCESS 2")
while(tot_steps<1000):
    for i in non_outliers:
        temp = -2*(i-A)
        sum += temp
    learning_rate = 0.0001
    step_size = learning_rate * sum
    A = A - step_size
    temp =0
    sum=0
    print(A, "@ step",tot_steps)
    tot_steps +=1

'''
'''
'''


#problem 3
D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/ramp_noise.xlsx')
list2 = D['x'].tolist()
len1 = len(list2)
list1 = []
for i in range(len1):
    list1.append([(D['x'].tolist())[i], (D['y'].tolist())[i]])
#len1 = len(list1)

#initial guess
slope = 1
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
    
'''
'''
'''
D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/sinusoid_noise.xlsx')
list2 = D['X'].tolist()
len1 = len(list2)
list1 = []
for i in range(len1):
    list1.append([(D['X'].tolist())[i], (D['y'].tolist())[i]])


#formula = (y- Acos((2*pi*n/100) + (pi*PHI/180)) )^2
#So d/dA --> 2*((y- Acos((2*pi*n/100) + (pi*PHI/180)) ) * cos((2*pi*n/100) + (pi*PHI/180))
#And, d/dPHI --> (y- Acos((2*pi*n/100) + (pi*PHI/180)) ) *(-Asin((2*pi*n/100)) + (pi*PHI/180))*(pi/180)


#initial guess
Amp = 1
PHI = 0

tot_steps = 0
Amp_sum=0
PHI_sum = 0

while(tot_steps<2000):
    for item in list1:
        temp1 = 2*(item[1]- Amp *np.cos((2*math.pi*item[0]/100) + (math.pi*PHI/180)) ) * (-1)*np.cos((2*math.pi*item[0]/100) + (math.pi*PHI/180))
        Amp_sum += temp1
        temp2 = 2*(item[1]- Amp *np.cos((2*math.pi*item[0]/100) + (math.pi*PHI/180)) ) *np.sin((2*math.pi*item[0]/100) + (math.pi*PHI/180))*(math.pi/180)
        PHI_sum += temp2
    learning_slope = 0.001
    learning_int = 0.01
    Amp  = Amp - Amp_sum* learning_slope
    PHI= PHI - PHI_sum*learning_int
    tot_steps +=1
    print(Amp,PHI, tot_steps)
    Amp_sum = 0
    PHI_sum = 0
    
'''




