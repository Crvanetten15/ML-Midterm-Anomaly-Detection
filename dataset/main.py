import sys
import csv
from typing import NoReturn
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt, log

from pandas import array


with open('./train/training-data.csv', "r") as t:
    tr = list(csv.reader(t, delimiter=" "))

headers = tr.pop(0)
headers = headers[0].split(',')
# print(headers)

for _ in range(len(tr)):
    tr[_] = tr[_][0].split(',')
    
NormalizeTr = np.array(tr, dtype=float).T

means = []

for i in range(1,27):
    sum = 0
    for _ in NormalizeTr[i]:
        sum += _ 
    means.append((sum/(len(NormalizeTr[i]))))  

# print(len(means))
vars = []

for x in range(1,27):
    top_sum = 0
    for _ in NormalizeTr[x]:
        top_sum += pow((_ - means[x-1]), 2)
    vars.append(sqrt((top_sum/(len(NormalizeTr[i])))))    
m_temp = list(means)
# for i in range(9797):
#     means.append(m_temp)
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
    temp = []
    indexes = [2, 3, 19]
    [temp.append(x[i]) for i in indexes]
    x = np.array(temp)

    # x = np.array(x, dtype=float)
    # left part of equation needed 
    print(len(x[0]))
    val_left = 1 / (sqrt((2*pi))*var)
    # prin(val_left)

    # x's minus mu
    for attr in range(len(x)):
        for row in range(len(x[0])):
            x[attr][row] -= u[attr]
    

    # x set to squared, and bottom handled
    x = (pow(x, 2)) 
    var = (2 * pow(var,2))
    # print(x[0][0])

    # Does parathesis math
    for att in range(len(x)): # 26
        for ro in range(len(x[0])):
            x[att][ro] /= var[att]
    # print(x[0][0])
    x = x * -1
 
    x = np.exp(x)

    for att in range(len(x)):
        for ro in range(len(x[0])):
            x[att][ro] *= val_left[att]

    # good to here
 
    x = x.T
# x   y row
# attr| | | | | | | | |
# 1   | | | | | | | | |
# 2   | | | | | | | | |
#     | | | | | | | | |
#     | | | | | | | | |
#     | | | | | | | | |
#     | | | | | | | | |
#     | | | | | | | | |
#     | | | | | | | | |
    # handle the power sum


    values = []
    for _ in range(len(x)):
        values.append(np.prod(x[_]))

    values = np.array(values,  dtype = float)
    
    print(values)
    
    summ = np.sum(values)

    return summ



'''
Task 3 - Below
'''
epsilon = 2.0

# print(np.linalg.det(cov_data))
# exit()
def PredictionMulti(CvMat, d, x, u):
    x = np.array(x, dtype=float)
    val_left = 1 / ((pow((2*pi), (d/2))) * (pow(np.linalg.det(CvMat), (1/2)))) 

    for attr in range(len(x)):
        for row in range(len(x[0])):
            x[attr][row] -= u[attr]    


    val_right = np.exp((-1/2)*(x-u)*(np.linalg.inv(CvMat))* (x - u))

    full_value = val_left * val_right
    return full_value

# print(Prediction(cov_data, 5, , means))
# IndexUsed = [5,6,7,14,15]
# NormalizeTr = NormalizeTr[1:]
# epsilonVal = PredictionIndependent(NormalizeTr, vars, means)
# print(epsilonVal)

validInd = []
for _ in range(23):
    with open(f'./valid/{_}.csv', "r") as t:
        temp = list(csv.reader(t, delimiter=" "))
    for d in range(len(temp)):
        temp[d] = temp[d][0].split(',')
    temp.pop(0)
    temp = np.array(temp, dtype=float).T
    temp = temp[1:]# 0th index is time
    
    means = []
    vars = []

    for u in range(len(temp)):
        means.append(np.mean(temp[u]))
        vars.append(np.std(temp[u]))

        

    # for i in range(9797):
    #     means.append(m_temp)
    means = np.array(means, dtype=float)
    vars = np.array(vars, dtype=float)

    print(means)
    print(vars) 

    validInd.append(PredictionIndependent(temp, vars, means))

    # PredictionMulti(temp, 5,vars, means)









ValidKey = [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]
for s in range(23):
    if validInd[s] >= 2.22e-12:
        validInd[s] = 1

NPs = [0,0,0,0] # TP, FP, TN, FN

for _ in range(len(validInd)):
    # print(validInd[_], ValidKey[_])
    if validInd[_] == ValidKey[_]:
        if validInd[_] == 1:
            NPs[0] += 1
        else:
            NPs[2] += 1
    else:
        if validInd[_] == 1:
            NPs[1] += 1
        else:
            NPs[3] += 1

PREC = NPs[0] / (NPs[0] + NPs[1])
REC = NPs[0] / (NPs[0] + NPs[3])
F1 = 2 * (PREC * REC) / (PREC + REC)

print(f'TP: {NPs[0]}FP: {NPs[1]} TN: {NPs[2]}FN: {NPs[3]}')
print("Precision: {}".format(PREC))
print("Recall: {}".format(REC))
print("F1: {}".format(F1))

    # for i in range(len(tr)):
    #     temp[i] = temp[i][0].split(',')
    #     print(tr[0])
    # for x in IndexUsed:
    #     print(f'{PredictionMulti(cov_data, 5, temp[x], means)}')
    # exit()