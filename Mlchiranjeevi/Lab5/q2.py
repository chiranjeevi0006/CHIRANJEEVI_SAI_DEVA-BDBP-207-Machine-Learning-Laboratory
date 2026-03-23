#sigmoid
import numpy as np
import matplotlib.pyplot as plt

x=np.array([[27,59,78],[43,63,94],[24,62,85],[23,75,83],[12,36,94]])
theta=np.array([[2],[2],[2]])
z=x @ theta
def sig_moid(A):
    sing=1/(1+np.exp(-A))
    return sing
print(sig_moid(z))
sing=sig_moid(x)
plt.plot(x,sing)
plt.xlabel("z")
plt.ylabel("g(z)")
plt.show()