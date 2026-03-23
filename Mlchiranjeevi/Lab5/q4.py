import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

X = df.drop(["diagnosis","id","Unnamed: 32"], axis=1)
y = df["diagnosis"]

x = X.values
y = y.values


le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.70,random_state=999)

aggregate = StandardScaler()
x_train_scaled = aggregate.fit_transform(x_train)
x_test_scaled = aggregate.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled,y_train)

y_predict = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_predict)
print("Accuracy of the model is :", accuracy)
