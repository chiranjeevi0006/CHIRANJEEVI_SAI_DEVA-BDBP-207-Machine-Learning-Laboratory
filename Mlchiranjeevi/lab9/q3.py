import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("sonar data.csv")
x = data.iloc[:, :-1]
y=data["R"]
print(y)
le = LabelEncoder()
y = le.fit_transform(y)\


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)

model=DecisionTreeRegressor(random_state=43)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=x_train.columns,filled=True)
plt.show()