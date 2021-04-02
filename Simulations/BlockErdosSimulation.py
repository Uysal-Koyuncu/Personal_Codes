import numpy as np
import matplotlib.pyplot as plt
"""
Block Stochastic Erdos Model
"""
n = 100
blockCount = 5
col = np.zeros(n)
p = np.random.random_sample((blockCount, 1))
print(p)
erd = np.empty([n, 1])
for b in range(blockCount):
    for i in range((int) (n / blockCount)):
        rancol = np.random.random_sample((n, 1))
        for j in range(n):
            if  j < n/blockCount * b or j >= n/blockCount * (b+1) or rancol[j][0] >= p[b]:
                rancol[j][0] = 0
            else:
                rancol[j][0] = 1
        colsum = np.sum(rancol)
        if colsum != 0:
            col = rancol / colsum
        erd = np.hstack((erd, col))

erd = np.transpose(erd)
erd = erd[1:]
erd = np.transpose(erd)
print(erd)
# plt.imshow(erd)
# erdInv = np.linalg.pinv(erd)
C = np.linalg.pinv(np.identity(n) - erd)
plt.imshow(C)
plt.colorbar()
plt.show()

