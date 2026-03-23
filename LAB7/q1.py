import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main():
    data=pd.read_csv("sonar data.csv")
    x = data.iloc[:, :-1]
    y=data["R"]
    print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    # y_new = []
    #
    # for i in y:
    #     if i == 'M':
    #         y_new.append(1)
    #     else:
    #         y_new.append(0)
    #
    # y = np.array(y_new)
    # print(y)
    x_split = np.array_split(x, 10)
    y_split = np.array_split(y, 10)
    k = 10
    for i in range(k):
        x_test = x_split[i]
        y_test = y_split[i]
        x_train = np.concatenate(x_split[:i] + x_split[i + 1:])
        y_train = np.concatenate(y_split[:i] + y_split[i + 1:])

        aggregate = StandardScaler()
        aggregate.fit(x_train)
        x_train_scaled = aggregate.transform(x_train)
        x_test_scaled = aggregate.transform(x_test)

        model = LogisticRegression()
        model.fit(x_train_scaled, y_train)

        y_predict = model.predict(x_test_scaled)

        accuracy = np.mean(y_predict == y_test)
        print("Accuracy of each fold:", accuracy)
if __name__ == "__main__":
    main()