import pandas as pd
import numpy as np

# read csv
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = df.drop(columns=["disease_score_fluct"]).values #x contains all the input
y = df["disease_score_fluct"].values.reshape(-1,1) # y has the output we need to predict and we reshaped to cplumn vector

# feature scaling is done, this helps all features are same range
#this helps gradient descent to converge faster
X = (X - X.mean(axis=0)) / X.std(axis=0)

#here we ll add an column of ones to include bias term theta0
#whis helps to handle intercept in the matrix form
ones = np.ones((X.shape[0],1))
X = np.hstack((ones, X))
                         #------------------------
k = len(y)               #k is the number of training
alpha = 0.01             #alpha  is learning rate
iterations = 1000        #
theta = np.zeros((X.shape[1],1)) #initially theta is zeros

# gradient descent
for i in range(iterations):

    y_pred = X.dot(theta)

    differ = y_pred - y   #here the differ is the difference btw predicted value and actual value

    cost = (1/(2*k)) * np.sum(differ**2)

    gradient = (1/k) * X.T.dot(differ)

    theta = theta - alpha * gradient

    if i % 100 == 0:
        print("iteration", i, "cost", cost)

print("final theta values")
print(theta)
#here we used that liner algebra to find inverse
theta_ne = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print("normal equation theta")
print(theta_ne)




