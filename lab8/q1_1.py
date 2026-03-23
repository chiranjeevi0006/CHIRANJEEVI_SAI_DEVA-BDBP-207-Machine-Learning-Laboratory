# L2 norm preprocessing example
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder



def main():

    data = pd.read_csv("breast-cancer.csv")
    print(data)

    # Features and target
    x = data.iloc[:, :9]
    y = data.iloc[:, 9]

    print(x)
    print(y)

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=800
    )

    # -------------------------
    # Ordinal Encoding
    # -------------------------
    x_ordi = x_train.iloc[:, [0, 2, 3]]

    x_ordii = pd.DataFrame()

    for col in x_ordi.columns:
        ordi = OrdinalEncoder()
        x_ordii[col] = ordi.fit_transform(x_ordi[[col]]).flatten()

    print("Ordinal Encoded Data:")
    print(x_ordii)

    # -------------------------
    # Label Encoding
    # -------------------------
    x_label = x_train.iloc[:, [4, 6, 8]].copy()

    for col in x_label.columns:
        le = LabelEncoder()
        x_label[col] = le.fit_transform(x_label[col])

    print("Label Encoded Data:")
    print(x_label)

    # -------------------------
    # One Hot Encoding
    # -------------------------
    X_onehot = x_train.iloc[:, [1, 5, 7]].copy()
    X_onehot.columns = X_onehot.columns.astype(str)

    OHE = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    OH_array = OHE.fit_transform(X_onehot)

    OH_col_names = OHE.get_feature_names_out(X_onehot.columns)

    X_onehot = pd.DataFrame(OH_array, columns=OH_col_names)

    print("One Hot Encoded Data:")
    print(X_onehot.head())


if __name__ == "__main__":
    main()