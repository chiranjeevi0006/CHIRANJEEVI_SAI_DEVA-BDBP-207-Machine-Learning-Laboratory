from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd


def main():
    data_frame=pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    # print(data_frame)
    # print(data_frame_y)

    x=data_frame.drop(["disease_score","disease_score_fluct"],axis=1)
    y=data_frame["disease_score_fluct"]

    # print(x)
    # print(y)
    print("--------------------------------------------------")
    # print(data_frame.shape)
    x_train, x_test, y_train, y_test=train_test_split(x,y ,test_size=0.3,random_state=700)
    # print(x_train)
    # print(x_test)
    # print(y_train)
    # print(y_test)
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    # print(x_test)
    print("-------------------------------------------")
    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred_train=model.predict(x_train)
    y_pred_test=model.predict(x_test)
    # print(y_pred_train)
    print("---------------------------------------------------")
    print("Train R2:", r2_score(y_train, y_pred_train))
    print("Test R2:", r2_score(y_test, y_pred_test))

if __name__ == "__main__":
        main()