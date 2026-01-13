import numpy as np
import matplotlib.pyplot as plt
mu = 0
sigma = 15
start = -100
stop = 100
num = 100

x = np.linspace(start, stop, num)

pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# plot
plt.plot(x, pdf)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Gaussian PDF (μ = 0, σ = 15)")
plt.show()
