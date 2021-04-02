import numpy as np
import matplotlib.pyplot as plt

n = 100
result = np.zeros((n, n))

for i in range(300):
    k = np.random.uniform(0, 1, (n, n))
    i = np.identity(n)
    m = i - k
    inv = np.linalg.inv(m)
    result = np.add(result, inv)

result[result < 0] = 0
result[result > 300] = 300

plt.imshow(result)
plt.colorbar()
plt.show()


