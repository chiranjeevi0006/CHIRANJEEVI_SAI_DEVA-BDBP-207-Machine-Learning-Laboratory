import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
def main():

    data = pd.read_csv("data.csv")
    X = data.drop(["diagnosis","id","Unnamed: 32"],axis=1)
    y = data["diagnosis"]
    X = X.values
    y = y.values

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_split = np.array_split(X,10)
    y_split = np.array_split(y,10)
    k = 10


    for i in range(k):
        x_test = x_split[i]
        y_test = y_split[i]
        x_train = np.concatenate(x_split[:i] + x_split[i+1:])
        y_train = np.concatenate(y_split[:i] + y_split[i+1:])

        aggregate = StandardScaler()
        aggregate.fit(x_train)
        x_train_scaled = aggregate.transform(x_train)
        x_test_scaled = aggregate.transform(x_test)

        model = LogisticRegression()
        model.fit(x_train_scaled,y_train)

        y_predict = model.predict(x_test_scaled)

        accuracy = accuracy_score(y_test,y_predict)     
        print("Accuracy of each fold:", accuracy)

if __name__ == "__main__":
    main()