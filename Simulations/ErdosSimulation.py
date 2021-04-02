import numpy as np
import matplotlib.pyplot as plt
from RandomPartitionErdos import randomPartitionErdos
"""
Erdos Model
"""

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
# should be reduced to 200->1000 round?
p = 0.30

# erd = simpleErdos(200, 0.30)
# print(erd)
# plt.imshow(erd)
# plt.colorbar()
# plt.show()
# C = np.linalg.pinv(np.identity(n) - erd) - np.identity(n)
runCount = 500
record = np.zeros((runCount, n, n))
for t in range(runCount):
    erd = randomPartitionErdos(n, p, 0.9)
    res = np.linalg.pinv(np.identity(n) - erd) - np.identity(n)
    record[t] = res

print(record[0])

expectRes = np.linalg.pinv(np.identity(n) - constantMat(n, p, 0.9)) - np.identity(n)
print(expectRes)

firstRowSums = np.zeros(runCount)
for i in range(runCount):
    firstRow = record[i][0]
    firstRowSums[i] = np.sum(firstRow)

firstRowSums = firstRowSums - 9
print(firstRowSums)
plt.plot(firstRowSums)
plt.show()
# But here it is just 0.9 in the avg case? What happens if there are different blocks

# varTable = np.zeros((n, n))
# for i in range(n):
#     for j in range(n):
#         vararr = np.zeros(runCount)
#         for k in range(runCount):
#             vararr[k] = record[k][i][j]
#         varTable[i][j] = np.mean(vararr)
#         # print(vararr)
#         # print(np.std(vararr))

# rowsum = np.sum(varTable, axis=1)
# print(np.std(rowsum)[0])
# plt.plot(rowsum)
# plt.show()

# Look at it instance-wise. Maybe some non-negative statistics (std?) recorded for each time, and then
# look at how the result look like.
"""
TODO: Check the sum of the rows
Also check the sum with the "constant matrix's inversion"
"""

# print(varTable)
# plt.imshow(varTable)
# plt.colorbar()
# plt.show()
# # print(erd)

# print(C)
# # plt.imshow(C)
# plt.imshow(C - np.identity(n))
