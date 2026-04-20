from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
def main():
    data=load_diabetes()
    print(data)
    x,y=data.data,data.target
    print(x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    model=RandomForestRegressor(n_estimators=100,random_state=0)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))



    data = load_iris()
    x, y = data.data, data.target


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)


    y_pred = model.predict(x_test)


    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
if __name__ == "__main__":
    main()