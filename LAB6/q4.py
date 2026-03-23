import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor


def main():
    data = pd.read_csv("data.csv")
    X = data.drop(["diagnosis", "id", "Unnamed: 32"], axis=1)
    y = data["diagnosis"]
    X = X.values
    y = y.values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=800)

    x_split = np.array_split(X_train, 10)
    y_split = np.array_split(y_train, 10)
    k = 10

    aggregate = StandardScaler()
    aggregate.fit(x_split)
    x_train_scaled = aggregate.transform(x_split)

    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)
    print(model)

    model2 = DecisionTreeRegressor()
    model2.fit(x_train_scaled, y_train)
    print(model2)
if __name__ == "__main__":
    main()