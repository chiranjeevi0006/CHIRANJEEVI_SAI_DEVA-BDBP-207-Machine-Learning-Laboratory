from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
x = pd.DataFrame(data.data)
y = pd.Series(data.target)   # FIX: use Series

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=7010
)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, leaf=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf = leaf

class CustomDecisionTreeClassifier:
    def entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def ig(self, x, y):
        best_ig = -1
        best_feature = None
        best_threshold = None

        root_entropy = self.entropy(y)
        n = len(y)

        for col in x.columns:
            values = np.sort(x[col].unique())
            thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

            for t in thresholds:
                left = y[x[col] <= t]
                right = y[x[col] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                ent = (len(left)/n)*self.entropy(left) + (len(right)/n)*self.entropy(right)
                ig = root_entropy - ent

                if ig > best_ig:
                    best_ig = ig
                    best_feature = col
                    best_threshold = t

        return best_feature, best_threshold

    def fit(self, x, y, min_samples=5):
        if len(y) < min_samples or len(y.unique()) == 1:
            return Node(leaf=y.mode()[0])

        feature, threshold = self.ig(x, y)

        if feature is None:
            return Node(leaf=y.mode()[0])

        node = Node(feature=feature, threshold=threshold)

        left_mask = x[feature] <= threshold
        right_mask = x[feature] > threshold

        node.left = self.fit(x[left_mask], y[left_mask], min_samples)
        node.right = self.fit(x[right_mask], y[right_mask], min_samples)

        self.tree = node
        return node

    def predict_one(self, row, node):
        if node.leaf is not None:
            return node.leaf

        if row[node.feature] <= node.threshold:
            return self.predict_one(row, node.left)
        else:
            return self.predict_one(row, node.right)

    def predict(self, x):
        return np.array([self.predict_one(row, self.tree) for _, row in x.iterrows()])


dt = CustomDecisionTreeClassifier()
dt.fit(x_train, y_train)


y_pred = dt.predict(x_test)


print("Accuracy:", accuracy_score(y_test, y_pred))