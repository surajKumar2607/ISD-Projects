import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

'''

D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/DC_noise1.xlsx')
list1 = D['y'].tolist()
sum =0
count = 0
for i in list1:
    sum += i
    count += 1

DC_val = sum/count
print (DC_val)
'''


'''
#2nd Problem
D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/DC_noise2.xlsx')
list2 = D['y'].tolist()
list2.sort()
sum2 =0
count2 =0

for i in list2:
    
    count2 += 1

Q1 = (list2[int(count2/4)])
Q3= (list2[int(3*count2/4)])

iqr = Q3-Q1

val1 = Q1 -1.5*iqr
val2 = Q3 + 1.5*iqr

list4 =[]

for i1 in list2:
    if((i1 > val1) and (i1<val2)):
        list4.append(i1)

len1 = len(list4)

sum =0
count = 0
for i in list4:
    sum += i
    count += 1

DC_val = sum/count
print (DC_val)

print(len1)

'''
'''

#problem 3
D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/ramp_noise.xlsx')
list1 = D['y'].tolist()
len1 = len(list1)

H= []

for i in range(len1):
    H.append([i, 1])

H = np.array(H)

Ht = H.transpose()

res1 = np.dot(Ht, H)

M1 = np.linalg.inv(res1)
list2 = np.array(list1)
res2 = np.dot(Ht, list2 )
result = M1 @ res2
print(result)

'''

'''
#Problem 4 sinusoidal

D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/sinusoid_noise.xlsx')
list1 = D['y'].tolist()
len1 = len(list1)

H= []

for i in range(0,len1,1):
    H.append([np.cos(math.pi*2*i/100), np.sin(2*math.pi*i/100)])

H = np.array(H)
Ht = H.transpose()
res1 = np.dot(Ht, H)

M1 = np.linalg.inv(res1)
list2 = np.array(list1)
res2 = np.dot(Ht, list2 )
result = M1 @ res2


amp = np.sqrt((result[0]*result[0] + result[1]*result[1]))
phase1 = np.arctan(-result[1]/result[0])
phase_in_deg = phase1*180/math.pi

print(amp, phase_in_deg)


'''

'''
#Problem 5 Piece Wise Linear

D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/noisy_signal.xlsx')
list1 = D['y'].tolist()
list1 = np.array(list1)
len1 = len(list1)

list2 =[]
for j in range(len1-1):
    list2.append(abs(list1[j+1]-list1[j]))

#no. of partition as per graph
partition = 5

len2 = len(list2)
list2.sort()
limit = (list2[len2-partition+1])

filteredsig =[]

sum =0
count =0
i = 0
index =0
while(i< len1):
    while ((i< len1-1) and (abs(list1[i]-list1[i+1])<limit)):
        sum += list1[i]
        count += 1
        i= i+1
        index +=1
    print(sum/count," at index =", index)
    for k in range(count+1):
        filteredsig.append(sum/count)
    sum =0
    count =0
    i=i+1
    index +=1

plt.plot(list1)
plt.plot(filteredsig)
plt.show()

'''

#Problem 5 Piece Wise Linear

D= pd.read_excel('C:/Users/91801/Documents/Personal Docs/MTECH/IIT Kgp Course Sem 2/6. ISD lab/noisy_signal.xlsx')
list1 = D['y'].tolist()
list1 = np.array(list1)
len1 = len(list1)

#formula to optimize = (Y-X)^2 + lam* (AX)^2
T = []


I = -1*np.identity(len1, dtype=int)

for i in range (1, len1):
    if (i==2):
        I[0][i]=0

for i in range(1,len1):
    I[i][i-1]= 1

I1 = np.identity(len1, dtype=int)

It = np.transpose(I)
Imul = It @ I
lam = 5

Mat1 = I1 + lam * Imul
Mat2 = np.linalg.inv(Mat1)

print(Mat2)

Ysol = Mat2 @ list1


plt.plot(Ysol)
plt.plot(list1)
plt.show()










