import numpy as np


theta = np.array([4, 2, 3, 3])


X = np.array([
    [1, 1, 2, 3],
    [1, 0, 1, 1],
    [1, 2, 0, 1],
    [1, 3, 2, 0],
    [1, 1, 1, 1]
])

y = X @ theta

print("Xθ =")
print(y)
