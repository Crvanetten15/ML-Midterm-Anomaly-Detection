import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


with open('./train/training-data.csv', "r") as t:
    tr = list(csv.reader(t, delimiter=" "))

headers = tr.pop(0)
headers = headers[0].split(',')

for _ in range(len(tr)):
    tr[_] = tr[_][0].split(',')
    
NormalizeTr = np.array(tr, dtype=float).T

# print(NormalizeTr)



# print(tr[0])
# print(headers)

figure, axis  = plt.subplots(3,9)

print(len(headers[1:]))

for x in range(9):
    print(x+1)
    axis[0, x].scatter(NormalizeTr[0], NormalizeTr[x+1])
    axis[0, x].set_title(f'{headers[x+1]}')

for x in range(9):
    print(x+10)
    axis[1, x%9].scatter(NormalizeTr[0], NormalizeTr[x+10])
    axis[1, x].set_title(f'{headers[x+10]}')

for x in range(9):
    print(x+19)
    if (x+19) > 25:
        break
    axis[2, x].scatter(NormalizeTr[0], NormalizeTr[x+19])
    axis[2, x].set_title(f'{headers[x+19]}')

# axis[0,0].scatter(NormalizeTr[0], NormalizeTr[1])
# # axis[0,0].scatter(normal)
# axis[0,0].set_title("f1_c vs T")

# axis[0,1].scatter(NormalizeTr[0], NormalizeTr[2])
# # axis[0,0].scatter(normal)
# axis[0,1].set_title("f1_a vs T")

# axis[0,2].scatter(NormalizeTr[0], NormalizeTr[3])
# # axis[0,0].scatter(normal)
# axis[0,2].set_title("f1_s vs T")

# axis[0,3].scatter(NormalizeTr[0], NormalizeTr[4])
# # axis[0,0].scatter(normal)
# axis[0,3].set_title("f1_d vs T")

# axis[0,4].scatter(NormalizeTr[0], NormalizeTr[5])
# # axis[0,0].scatter(normal)
# axis[0,4].set_title("f2_c vs T")

# axis[0,5].scatter(NormalizeTr[0], NormalizeTr[6])
# # axis[0,0].scatter(normal)
# axis[0,5].set_title("f2_a vs T")

# axis[0,6].scatter(NormalizeTr[0], NormalizeTr[7])
# # axis[0,0].scatter(normal)
# axis[0,6].set_title("f2_s vs T")

# axis[0,7].scatter(NormalizeTr[0], NormalizeTr[8])
# # axis[0,0].scatter(normal)
# axis[0,7].set_title("f2_d vs T")

# axis[0,8].scatter(NormalizeTr[0], NormalizeTr[9])
# # axis[0,0].scatter(normal)
# axis[0,8].set_title("prg_c vs T")

# axis[1,0].scatter(NormalizeTr[0], NormalizeTr[10])
# # axis[0,0].scatter(normal)
# axis[1,0].set_title("prg_a vs T")

# axis[1,1].scatter(NormalizeTr[0], NormalizeTr[11])
# # axis[0,0].scatter(normal)
# axis[1,1].set_title("prg_s vs T")

# axis[1,2].scatter(NormalizeTr[0], NormalizeTr[12])
# # axis[0,0].scatter(normal)
# axis[1,2].set_title("prg_d vs T")

# axis[1,3].scatter(NormalizeTr[0], NormalizeTr[13])
# # axis[0,0].scatter(normal)
# axis[1,3].set_title("prd_c vs T")

# axis[1,4].scatter(NormalizeTr[0], NormalizeTr[14])
# # axis[0,0].scatter(normal)
# axis[1,4].set_title("prd_a vs T")

# axis[1,5].scatter(NormalizeTr[0], NormalizeTr[15])
# # axis[0,0].scatter(normal)
# axis[1,5].set_title("prd_s vs T")

# axis[1,6].scatter(NormalizeTr[0], NormalizeTr[16])
# # axis[0,0].scatter(normal)
# axis[1,6].set_title("prd_d vs T")

# axis[1,7].scatter(NormalizeTr[0], NormalizeTr[17])
# # axis[0,0].scatter(normal)
# axis[1,7].set_title("pr_s vs T")

# axis[1,8].scatter(NormalizeTr[0], NormalizeTr[18])
# # axis[0,0].scatter(normal)
# axis[1,8].set_title("pr_d vs T")

# axis[2,0].scatter(NormalizeTr[0], NormalizeTr[19])
# # axis[0,0].scatter(normal)
# axis[2,0].set_title("lq_s vs T")

# axis[2,1].scatter(NormalizeTr[0], NormalizeTr[20])
# # axis[0,0].scatter(normal)
# axis[2,1].set_title("cmp_a_s vs T")

# axis[2,2].scatter(NormalizeTr[0], NormalizeTr[21])
# # axis[0,0].scatter(normal)
# axis[2,2].set_title("cmp_a_d vs T")

# axis[2,3].scatter(NormalizeTr[0], NormalizeTr[22])
# # axis[0,0].scatter(normal)
# axis[2,3].set_title("cmp_b_s vs T")

# axis[2,4].scatter(NormalizeTr[0], NormalizeTr[23])
# # axis[0,0].scatter(normal)
# axis[2,4].set_title("cmp_b_d vs T")

# axis[2,5].scatter(NormalizeTr[0], NormalizeTr[24])
# # axis[0,0].scatter(normal)
# axis[2,5].set_title("cmp_c_s vs T")

# axis[2,6].scatter(NormalizeTr[0], NormalizeTr[25])
# # axis[0,0].scatter(normal)
# axis[2,6].set_title("cmp_c_d vs T")


plt.show()