import sys
import csv
from typing import NoReturn
from matplotlib.font_manager import list_fonts
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
Use of training data
'''
with open(f'./train/training-data.csv', "r") as t:
        trd = list(csv.reader(t, delimiter=" "))
for d in range(len(trd)):
    trd[d] = trd[d][0].split(',')
trd.pop(0)
trd = np.array(trd, dtype=float).T
trd = trd[1:]# 0th index is time
trd = np.array([trd[2],trd[3], trd[19]]) #[trd[2],trd[3], trd[19] 
means = []
std = []

for u in range(len(trd)):
    means.append(np.mean(trd[u]))

for i in range(len(trd)):  
    std.append(np.std(trd[i]))

means = np.array(means, dtype=float)
std = np.array(std, dtype=float)

def findTrEpsilon():
    return PredictionIndependent(trd.T)

def findTrEpsilonM():
    return PredictionMulti(trd.T, 3)
'''
End of training
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

# temp_p = (1 / (np.sqrt(2 * np.pi) * std[n])) * np.exp(-1 * ((_[n] - means[n]) ** 2) / (2 * std[n]))

def PredictionIndependent(CurrentMat):
    sum_of_prob = 0
    for _ in CurrentMat:
        p = 1
        for n in range(len(_)):
            
            left = (1 / (np.sqrt(2 * np.pi) * std[n]))
            right = np.exp(-1 * (pow((_[n] - means[n]), 2)) / (2 * std[n]))
            
            temp_p = left * right
            
            p *= temp_p
        # print(p)
        sum_of_prob += p

    return sum_of_prob

'''
Task 3 - Below
'''

'''
            |    CONSTANT VALUE    |     | each x is a vector of data sets | 
            ____________1__________
p(x;𝜇,Σ) = ((2𝜋)^(d/2))*(|Σ|^(1/2))    * exp((-1/2)*(x - 𝜇).T * (Σ^(-1)) * (x - 𝜇)))

'''
def PredictionMulti(x, d):
    totalsum = []
    CovMatrix = np.cov(x.T)
    CovMatrix = np.array(CovMatrix)
    list_of_right = []
    # print(len(x))
    for v in range(len(x)):
        val = np.linalg.det(CovMatrix)
        
        # if the codeterminate is 0 means the data is a 1 d array and no inveritable
        if val == 0:
            val = 0.000000000000000000001

        left = 1 / ((pow((2*pi), (d/2))) * (pow(val, (1/2)))) 
        sub_p= []
        for y in range(len(means)):
            sub_p.append(x[v][y]-means[y])
        
        sub_p = np.array(sub_p).reshape([len(sub_p), 1])
        
        # if the codetermine is 0 its not invertable and is even already 
        try:
            right = np.exp((-1/2) * sub_p.T @ (np.linalg.inv(CovMatrix)) @ sub_p )
        except:
            right = np.exp((-1/2) * sub_p.T @ CovMatrix @ sub_p )

        array = (left * right)
        # print(array)
        if (array == 0.0):
            list_of_right.append(0)
        
        list_of_right.append(1 if array < 6.6619532e-10 else 0)
    
    count = 0
    for _ in list_of_right:
            if _ == 1:
                count += 1


    return (count/len(list_of_right))


# Tracking variables for testing against valid
validMulti = []
validInd = []

for _ in range(23):

    with open(f'./valid/{_}.csv', "r") as t:
        temp = list(csv.reader(t, delimiter=" "))
    for d in range(len(temp)):
        temp[d] = temp[d][0].split(',')
    temp.pop(0)
    temp = np.array(temp, dtype=float).T
    temp = temp[1:]# 0th index is time
    temp = np.array([temp[2], temp[3], temp[19]])   # ,temp[5],temp[6], temp[14], temp[15]]) #,  temp[15]])

    validMulti.append(PredictionMulti(temp.T, 3))
    validInd.append(PredictionIndependent(temp.T))
    


epsilonInd =  findTrEpsilon()# 1.2e-06 # 8.5e-21
epsilonMulti =  .51 #5.09e-6#findTrEpsilonM()

def resetLists(validreport, epi):
    for e in range(23):
        if validreport[e] < epi: 
            validreport[e] = 1
        else:
            validreport[e] = 0


resetLists(validInd, epsilonInd)
resetLists(validMulti, epsilonMulti)

ValidKey = [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]

 # TP, FP, TN, FN
def FindAnomolies(table):
    NPs = [0,0,0,0]
    for _ in range(len(table)):
        if table[_] == ValidKey[_]:
            if table[_] == 1:
                NPs[0] += 1
            else:
                NPs[2] += 1
        else:
            if table[_] == 1:
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

'''
Valid Data reports
'''
print()
print('Independent - Valid\n',validInd)
FindAnomolies(validInd)

print()
print('MultiVariant - Valid\n',validMulti)
FindAnomolies(validMulti)

validMultiT = []
validIndT = []

for _ in range(58):

    with open(f'./test/{_}.csv', "r") as t:
        temp = list(csv.reader(t, delimiter=" "))
    for d in range(len(temp)):
        temp[d] = temp[d][0].split(',')
    temp.pop(0)
    temp = np.array(temp, dtype=float).T
    temp = temp[1:]# 0th index is time
    temp = np.array([temp[2], temp[3], temp[19]])   # ,temp[5],temp[6], temp[14], temp[15]]) #,  temp[15]])

    validMultiT.append(PredictionMulti(temp.T, 3))
    validIndT.append(PredictionIndependent(temp.T))


epsilonIndT =  findTrEpsilon()# 1.2e-06 # 8.5e-21
epsilonMultiT = .51#findTrEpsilonM()

def resetLists(validreport, epi):
    for e in range(58):
        if validreport[e] < epi: 
            validreport[e] = 1
        else:
            validreport[e] = 0


resetLists(validIndT, epsilonIndT)
resetLists(validMultiT, epsilonMultiT)

print('Inedependent Testing -\n', validIndT)
print('Multivariant Testing -\n', validMultiT)