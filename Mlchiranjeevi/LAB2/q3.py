from sklearn.datasets import fetch_california_housing
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