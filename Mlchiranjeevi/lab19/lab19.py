import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# LOAD DATA
data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

model = LogisticRegression(max_iter=1000)
model.fit(X,y)

y_pred = model.predict(X)

# METRICS
print("Accuracy:", accuracy_score(y,y_pred))
print("Precision:", precision_score(y,y_pred))
print("Recall:", recall_score(y,y_pred))
print("F1:", f1_score(y,y_pred))

# ROC
y_prob = model.predict_proba(X)[:,1]
fpr, tpr, _ = roc_curve(y, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.show()

print("AUC:", auc(fpr,tpr))