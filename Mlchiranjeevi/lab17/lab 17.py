import numpy as np
import matplotlib.pyplot as plt

# DATA
X = np.array([
[1,13],[1,18],[2,9],[3,6],[6,3],[9,2],[13,1],[18,1],
[3,15],[6,6],[6,11],[9,5],[10,10],[11,5],[12,6],[16,3]
])

labels = np.array(["Blue"]*8 + ["Red"]*8)

# -------- 2D PLOT --------
for i in range(len(X)):
    if labels[i] == "Blue":
        plt.scatter(X[i,0], X[i,1], color='blue')
    else:
        plt.scatter(X[i,0], X[i,1], color='red')

plt.title("Original Data")
plt.show()


# -------- TRANSFORM FUNCTION --------
def Transform(x):
    return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])


# -------- 3D TRANSFORM --------
X_trans = np.array([Transform(x) for x in X])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X_trans)):
    if labels[i] == "Blue":
        ax.scatter(*X_trans[i], color='blue')
    else:
        ax.scatter(*X_trans[i], color='red')

plt.title("3D Transformed Data")
plt.show()


# -------- DOT PRODUCT --------
x1 = np.array([3,6])
x2 = np.array([10,10])

phi_x1 = Transform(x1)
phi_x2 = Transform(x2)

dot_product = np.dot(phi_x1, phi_x2)
print("Dot Product in higher dimension:", dot_product)


# -------- POLYNOMIAL KERNEL --------
def poly_kernel(a,b):
    return a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2

print("Kernel Result:", poly_kernel(x1,x2))