import numpy as np
import matplotlib.pyplot as plt

# range
start = -10
stop = 10
num = 100

x1 = np.linspace(start, stop, num)

y = x1**2

# plot
plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = x1²")
plt.show()
