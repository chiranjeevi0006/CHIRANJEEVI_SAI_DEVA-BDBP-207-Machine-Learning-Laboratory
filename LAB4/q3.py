#(x transpose of x)-1 . x T y
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as lb


data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

x = data.drop(["disease_score_fluct"],axis=1)
y = data["disease_score_fluct"]

x=x.values
y=y.values

# print(data)


normal_eqn =np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),(np.dot(np.transpose(x),y)))
print(normal_eqn)
