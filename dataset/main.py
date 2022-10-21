import sys
import csv
from typing import NoReturn
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt


with open('./train/training-data.csv', "r") as t:
    tr = list(csv.reader(t, delimiter=" "))

headers = tr.pop(0)
headers = headers[0].split(',')
print(headers)

for _ in range(len(tr)):
    tr[_] = tr[_][0].split(',')
    
NormalizeTr = np.array(tr, dtype=float).T

means = []

for i in range(1,26):
    sum = 0
    for _ in NormalizeTr[i]:
        sum += _ 
    means.append((sum/(len(NormalizeTr[i]))))  

print(len(means))
vars = []

for x in range(1,26):
    top_sum = 0
    for _ in NormalizeTr[x]:
        top_sum += pow((_ - means[x-1]), 2)
    vars.append((top_sum/(len(NormalizeTr[i]))))    

means = np.array(means, dtype=float)
vars = np.array(vars, dtype=float)

# print(means)
# print(vars)
# exit()

'''
Creation of Graphs
'''
figure, axis  = plt.subplots(3,9)

for x in range(9):
    axis[0, x].scatter(NormalizeTr[0], NormalizeTr[x+1])
    axis[0, x].set_title(f'{headers[x+1]}')

for x in range(9):
    axis[1, x%9].scatter(NormalizeTr[0], NormalizeTr[x+10])
    axis[1, x].set_title(f'{headers[x+10]}')

for x in range(9):
    if (x+19) > 25:
        break
    axis[2, x].scatter(NormalizeTr[0], NormalizeTr[x+19])
    axis[2, x].set_title(f'{headers[x+19]}')

# plt.show()


'''
End of Graphs 
'''

'''
Pick the 'Normal' Data to create our Covariance Matrix
[0 - 'time', 
1 - 4 :     'f1_c', 'f1_a', 'f1_s', 'f1_d', 
5 - 8 :     'f2_c', 'f2_a', 'f2_s', 'f2_d',         5 6 7
9 - 12 :    'prg_c', 'prg_a', 'prg_s', 'prg_d', 
13 - 16 :   'prd_c', 'prd_a', 'prd_s', 'prd_d',     14 15
17 - 18 :   'pr_s', 'pr_d', 
19 - 20 :   'lq_s', 'lq_d',                         19
21 - 22 :   'cmp_a_s', 'cmp_a_d',                   21
23 - 24 :   'cmp_b_s', 'cmp_b_d',                   23
25 - 26 :   'cmp_c_s', 'cmp_c_d]                    25
'''

FeatureList = [NormalizeTr[5], NormalizeTr[6], NormalizeTr[7], NormalizeTr[14], NormalizeTr[15]]
# Covariant matrix made with 'f2_c', 'f2_a', 'f2_s', 'prd_a', 'prd_s' 
cov_data = np.cov(FeatureList)
# print(cov_data)

# equation now to find Guassian Distribution 
'''
d = number of features that you have 
    (1  / ((2 * PI) ^ (d/2) * |Σ| ^ 1/2)) *
'''
'''
Task 2 - below 
'''
def PredictionIndependent(x, var, u):
    # x = np.array(x, dtype=float)
    val_left = 1 / (sqrt((2*pi))*var)
    print(val_left)
    val_right = np.exp(-1 * (pow((x-u), 2)) / (2 * pow(var, 2)))

    full_value = val_left * val_right
    return full_value



'''
Task 3 - Below
'''
epsilon = 2.0


def PredictionMulti(CvMat, d, x, u):
    x = np.array(x, dtype=float)
    val_left = 1 / ((pow((2*pi), (d/2))) * (pow(np.linalg.det(CvMat), (1/2)))) 
    val_right = np.exp((-1/2)*(x-u).T*(np.linalg.inv(CvMat))* (x - u))

    full_value = val_left * val_right
    return full_value

# print(Prediction(cov_data, 5, , means))
# IndexUsed = [5,6,7,14,15]
for _ in range(23):
    with open(f'./valid/{_}.csv', "r") as t:
        temp = list(csv.reader(t, delimiter=" "))
    for _ in range(len(temp)):
        temp[_] = temp[_][0].split(',')
    temp.pop(0)
    temp = np.array(temp, dtype=float).T
    temp = temp[1:]# 0th index is time
    print(len(temp.T))
    # temp = temp.T
    PredictionIndependent(temp, vars, means)
    exit()







    # for i in range(len(tr)):
    #     temp[i] = temp[i][0].split(',')
    #     print(tr[0])
    # for x in IndexUsed:
    #     print(f'{PredictionMulti(cov_data, 5, temp[x], means)}')
    # exit()