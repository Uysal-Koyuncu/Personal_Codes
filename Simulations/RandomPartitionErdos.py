import numpy as np
import matplotlib.pyplot as plt

def randomPartitionErdos(n, p, c=0.9):
    col = np.zeros(n)
    erd = np.empty([n, 1])
    for i in range(n):
        rancol = np.random.random_sample((n, 1))
        for j in range(n):
            if rancol[j][0] >= p:
                rancol[j][0] = 0
            else:
                rancol[j][0] = np.random.rand(1)
        colsum = np.sum(rancol)
        col = rancol / colsum
        erd = np.hstack((erd, col))
    erd = np.transpose(erd)
    erd = erd[1:]
    erd = np.transpose(erd)

    return erd * c


n = 50
p = 0.30

erd = randomPartitionErdos(200, 0.30, 0.9)
plt.imshow(erd)
plt.colorbar()
plt.show()