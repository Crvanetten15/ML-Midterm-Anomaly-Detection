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
def findTrEpsilon():
    with open(f'./train/training-data.csv', "r") as t:
            trd = list(csv.reader(t, delimiter=" "))
    for d in range(len(trd)):
        trd[d] = trd[d][0].split(',')
    trd.pop(0)
    trd = np.array(trd, dtype=float).T
    trd = trd[1:]# 0th index is time
    trd = np.array([trd[7],trd[5], trd[6], trd[15], trd[14]]) #, temp[14], temp[15]])

    means = []
    std = []
        
    for u in range(len(trd)):
        means.append(np.mean(trd[u]))
        std.append(np.std(trd[u]))

    means = np.array(means, dtype=float)
    std = np.array(vars, dtype=float)

    return PredictionIndependent(trd.T, means, std)

def findTrEpsilonM():
    with open(f'./train/training-data.csv', "r") as t:
            trd = list(csv.reader(t, delimiter=" "))
    for d in range(len(trd)):
        trd[d] = trd[d][0].split(',')
    trd.pop(0)
    trd = np.array(trd, dtype=float).T
    trd = trd[1:]# 0th index is time
    trd = np.array([trd[7],trd[5], trd[6], trd[15], trd[14]]) #, temp[14], temp[15]])

    means = []
    std = []
        
    for u in range(len(trd)):
        means.append(np.mean(trd[u]))
        std.append(np.std(trd[u]))

    means = np.array(means, dtype=float)
    std = np.array(vars, dtype=float)
    return PredictionMulti(trd.T, means, 5)

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

# FeatureList = [NormalizeTr[5], NormalizeTr[6], NormalizeTr[7], NormalizeTr[14], NormalizeTr[15]]
# # Covariant matrix made with 'f2_c', 'f2_a', 'f2_s', 'prd_a', 'prd_s' 
# cov_data = np.cov(FeatureList)
# print(cov_data)

# equation now to find Guassian Distribution 
'''
d = number of features that you have 
    (1  / ((2 * PI) ^ (d/2) * |Σ| ^ 1/2)) *
'''


'''
Task 2 - below 
'''

def PredictionIndependent(x, mean, std):

    sum_of_prob = 0
    for _ in x:
        # print(_)
        current_p = 1
        for j in range(len(_)):
            left = (1 /(np.sqrt(2*pi)*std[j]))
            right = np.exp(-1*(pow((_[j] - mean[j]), 2))/(2 * std[j]))
            # print(left, right)
            temp_p = left * right
            # print(temp_p)
            current_p *= temp_p
        sum_of_prob += current_p
    return sum_of_prob



'''
Task 3 - Below
'''

# print(np.linalg.det(cov_data))
# exit()
def PredictionMulti(x, mean, d):

    CovMatrix = np.cov(x.T)

    # print(np.linalg.det(CovMatrix))

    try:
        left = 1 / ((pow((2*pi), (d/2))) * (pow(np.linalg.det(CovMatrix), (1/2)))) 
    # print(left)
        sub_p= np.array((x-mean))
    # print('hi', sub_p.dot(np.linalg.inv(CovMatrix)) * sub_p)

        right = np.exp( (-1/2) * sub_p.dot(np.linalg.inv(CovMatrix)) * sub_p )
    except:
        return 0
    # print(right)

    full_value = (left * right)
    temp = []
    for _ in range(len(full_value)):
        temp.append(np.prod(full_value[_]))
    print(np.sum(temp))
    return np.sum(temp)

# print(Prediction(cov_data, 5, , means))
# IndexUsed = [5,6,7,14,15]
# NormalizeTr = NormalizeTr[1:]
# epsilonVal = PredictionIndependent(NormalizeTr, vars, means)
# print(epsilonVal)

validInd = []

# print(findTrEpsilonM())
# exit()
for _ in range(23):
    # with open(f'./test/{_}.csv', "r") as t:
    with open(f'./valid/{_}.csv', "r") as t:
        temp = list(csv.reader(t, delimiter=" "))
    for d in range(len(temp)):
        temp[d] = temp[d][0].split(',')
    temp.pop(0)
    temp = np.array(temp, dtype=float).T
    temp = temp[1:]# 0th index is time
    temp = np.array([temp[7],temp[5], temp[6], temp[15], temp[14]]) #, temp[14], temp[15]])

    means = []
    std = []
    
    for u in range(len(temp)):
        means.append(np.mean(temp[u]))
        std.append(np.std(temp[u]))

    means = np.array(means, dtype=float)
    std = np.array(vars, dtype=float)
    # print(PredictionMulti(temp.T, means, 5))

   
    # print(var 
    validInd.append(PredictionIndependent(temp.T, means, vars))
    # validInd.append(PredictionMulti(temp.T, means, 5))
    # PredictionMulti(temp, 5,vars, means)


epsilon = findTrEpsilon() # 8.5e-21

for e in range(23):
    if validInd[e] > epsilon:
        validInd[e] = 1
    else:
        validInd[e] = 0

print(validInd)

# exit()

ValidKey = [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]
SubmissionKeyforFun = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]

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

