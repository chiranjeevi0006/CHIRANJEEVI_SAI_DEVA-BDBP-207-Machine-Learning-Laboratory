from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
def main():
    data = load_diabetes()
    X, y = data.data, data.target
    print(X)
    print(y)


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    base_model = DecisionTreeRegressor()

    bag_reg = BaggingRegressor(estimator=base_model,n_estimators=50,random_state=42)


    bag_reg.fit(X_train, y_train)


    y_pred = bag_reg.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)
    print("Bagging Regressor MSE:", mse)


if __name__ == "__main__":
    main()