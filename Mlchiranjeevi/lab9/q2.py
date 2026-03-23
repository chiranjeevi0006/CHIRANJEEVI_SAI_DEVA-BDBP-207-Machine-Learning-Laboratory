import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt


df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

x=df.drop("disease_score",axis=1)
y=df["disease_score"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)

model=DecisionTreeRegressor(random_state=43)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
plt.figure(figsize=(12,120))
plot_tree(model, feature_names=x_train.columns,filled=True)
plt.show()