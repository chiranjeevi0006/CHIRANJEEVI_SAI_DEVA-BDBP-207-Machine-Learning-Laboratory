import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# read csv
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = df.drop(columns=["disease_score_fluct"]).values #x contains all the input
y = df["disease_score_fluct"].values.reshape(-1,1) # y has the output we need to predict and we reshaped to cplumn vector
print(X.shape)
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
# theta_ne = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# print("normal equation theta")
# print(theta_ne)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def main():
    X, y = fetch_california_housing(return_X_y=True)
    # print(X)
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.3, random_state=500)
    # print(X_train,X_test)
    # print(y_train,y_test)


    scalar = StandardScaler()
    X_train= scalar.fit_transform(X_train)
    X_test=scalar.transform(X_test)
    # print(X_train)
    # print(X_test)

    reg=LinearRegression()
    reg.fit(X_train,y_train)
    print(X_train)
    print(y_train)

    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    print("Train R2:", r2_score(y_train, y_train_pred))
    print("Test R2:", r2_score(y_test, y_test_pred))
    print()
if __name__=="__main__":
    main()