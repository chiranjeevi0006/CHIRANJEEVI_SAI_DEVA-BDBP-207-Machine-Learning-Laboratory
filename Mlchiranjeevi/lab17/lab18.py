import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# DATA
X = np.array([
[6,5],[6,9],[8,6],[8,8],[8,10],[9,2],[9,5],[10,10],
[10,13],[11,5],[11,8],[12,6],[12,11],[13,4],[14,8]
])

y = np.array(["Blue","Blue","Red","Red","Red","Blue","Red","Red",
              "Blue","Red","Red","Red","Blue","Blue","Blue"])

# convert labels
y_num = np.array([0 if i=="Blue" else 1 for i in y])

# RBF SVM
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X, y_num)

# Polynomial SVM
model_poly = SVC(kernel='poly', degree=2)
model_poly.fit(X, y_num)

print("RBF Accuracy:", model_rbf.score(X,y_num))
print("Poly Accuracy:", model_poly.score(X,y_num))