import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# -------------------------------
# Hypothesis function
# h(x) = Xθ
# -------------------------------
def hypothesis(X, theta):
    return X @ theta


# -------------------------------
# Cost function (Mean Squared Error)
# J(θ) = 1/(2m) * Σ(h(x)-y)^2
# -------------------------------
def cost_function(X, y, theta):
    m = len(y)
    error = X @ theta - y
    cost = (1/(2*m)) * (error.T @ error)
    return float(cost)


# -------------------------------
# Add bias column
# -------------------------------
def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


# -------------------------------
# Gradient calculation
# ∇J(θ) = 1/m * Xᵀ(Xθ - y)
# -------------------------------
def gradient(X, y, theta):
    m = len(y)
    prediction = hypothesis(X, theta)
    error = prediction - y
    grad = (1/m) * (X.T @ error)
    return grad


# -------------------------------
# Main program
# -------------------------------
def main():

    # Load dataset
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    # Input features
    X = df.drop(["disease_score", "disease_score_fluct"], axis=1).values

    # Target variable
    y = df["disease_score"].values.reshape(-1,1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=800
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add bias column
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    # Initialize theta
    theta = np.zeros((X_train.shape[1],1))

    # Hyperparameters
    alpha = 0.1
    iterations = 1000

    # Gradient Descent
    for i in range(iterations):

        theta = theta - alpha * gradient(X_train, y_train, theta)

        if i % 100 == 0:
            cost = cost_function(X_train, y_train, theta)
            print("Iteration:", i, "Cost:", cost)

    # Predictions
    y_pred_train = hypothesis(X_train, theta)
    y_pred_test = hypothesis(X_test, theta)

    print("\n-----------------------------------")
    print("Final theta values:")
    print(theta)

    print("\nModel Performance")
    print("Train R2:", r2_score(y_train, y_pred_train))
    print("Test R2:", r2_score(y_test, y_pred_test))


# Run program
if __name__ == "__main__":
    main()