import numpy as np
import matplotlib.pyplot as plt
from RandomPartitionErdos import randomPartitionErdos

def simpleErdos(n, p, c):
    col = np.zeros(n)
    erd = np.empty([n, 1])
    for i in range(n):
        rancol = np.random.random_sample((n, 1))
        for j in range(n):
            if rancol[j][0] >= p:
                rancol[j][0] = 0
            else:
                rancol[j][0] = 1
        colsum = np.sum(rancol)
        col = rancol / colsum
        erd = np.hstack((erd, col))
    erd = np.transpose(erd)
    erd = erd[1:]
    erd = np.transpose(erd)

    return erd * c

def constantMat(n, p, c):
    res = np.ones((n,n)) /n
    return res * c

n = 100
p = 0.3
runCount = 500
record = np.zeros((runCount, n, n))
origin = np.zeros((runCount, n, n))
for t in range(runCount):
    erd = randomPartitionErdos(n, p, 0.9)
    res = np.linalg.pinv(np.identity(n) - erd) - np.identity(n)
    origin[t] = erd
    record[t] = res

print(origin[0])
print(record[0])

expectedOri = np.identity(n) - constantMat(n, p, 0.9)
print(expectedOri)
expectRes = np.linalg.pinv(expectedOri)
expectRes2 = np.linalg.inv(expectedOri)
print(expectRes)
print(expectRes2 @ expectedOri)

firstRowSums = np.zeros(runCount)
origin_firstRowSums = np.zeros(runCount)
for i in range(runCount):
    firstRow = record[i][0]
    firstRowSums[i] = np.sum(firstRow)
    ori_firstRow = origin[i][0]
    origin_firstRowSums[i] = np.sum(ori_firstRow)

origin_firstRowSums = origin_firstRowSums - 0.9 # A
firstRowSums = firstRowSums - 9 # (I-A)^-1
# Calculation: calculate all row sums, calculate the maximum distance between these,
# and then record this value for all the runs
ori_mean = np.mean(origin_firstRowSums)
ori_std = np.std(origin_firstRowSums)
res_mean = np.mean(firstRowSums)
res_std = np.std(firstRowSums)
print(ori_mean, ori_std) # A
print(res_mean, res_std) # (I-A)^-1
valid = [i for i in firstRowSums if (i > -res_std and i < res_std)]
print(len(valid))
plt.plot(firstRowSums)
plt.show()

