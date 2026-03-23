
import matplotlib.pynum as plt

X1 = []
Y = []

start = -100
stop = 100
num = 100
step = (stop - start) / (num - 1)

for i in range(num):
    x = start + i * step
    y = 2 * x + 3
    X1.append(x)
    Y.append(y)

plt.plot(X1, Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("y = 2x + 3")
plt.show()
