import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.DataFrame({
    "x_1": [2,1,3,4,5,6],
    "x_2": [3,2,1,3,2,1],
    "x_3": [1,2,2,3,1,2],
    "y_true": [5,4,6,9,10,12]
})

class ScratchDecisionTreeRegressor:
    def fit(self, X, y, min_samples=2):
        self.tree = self.build_tree(X, y, min_samples)
        return self

    def mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def best_split(self, X, y):
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for col in X.columns:
            values = np.sort(X[col].unique())
            thresholds = [(values[i] + values[i+1])/2 for i in range(len(values)-1)]

            for t in thresholds:
                left = y[X[col] <= t]
                right = y[X[col] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                mse_split = (
                    len(left)*self.mse(left) +
                    len(right)*self.mse(right)
                ) / len(y)

                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature = col
                    best_threshold = t

        return best_feature, best_threshold

    def build_tree(self, X, y, min_samples):
        if len(y) < min_samples or len(set(y)) == 1:
            return np.mean(y)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return np.mean(y)

        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold

        left_tree = self.build_tree(X[left_mask], y[left_mask], min_samples)
        right_tree = self.build_tree(X[right_mask], y[right_mask], min_samples)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def predict_one(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        if row[tree["feature"]] <= tree["threshold"]:
            return self.predict_one(row, tree["left"])
        else:
            return self.predict_one(row, tree["right"])

    def predict(self, X):
        return np.array([self.predict_one(row, self.tree) for _, row in X.iterrows()])


X = df[["x_1", "x_2", "x_3"]]
y = df["y_true"]

model = ScratchDecisionTreeRegressor()
model.fit(X, y)

pred = model.predict(X)

print("Predictions:", pred)
print("MSE:", mean_squared_error(y, pred))