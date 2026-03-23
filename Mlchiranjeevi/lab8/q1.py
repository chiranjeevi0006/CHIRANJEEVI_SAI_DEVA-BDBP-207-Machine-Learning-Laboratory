import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder


def l1_norm_dataset(X):
    result = []

    for row in X.values:

        s = 0
        for val in row:
            s = s + abs(val)

        result.append(s)

    return result


def l2_norm_dataset(X):
    result = []

    for row in X.values:

        s = 0
        for val in row:
            s = s + (val * val)

        result.append(math.sqrt(s))

    return result


def main():
    data = pd.read_csv("breast-cancer.csv")
    print("Original Data")
    print(data.head())

    X = data.iloc[:, :9]
    y = data.iloc[:, 9]

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=800
    )

    # -------- ORDINAL ENCODING --------
    X_ordi = X_train.iloc[:, [0, 2, 3]]

    ord_enc = OrdinalEncoder()

    X_ordi_enc = pd.DataFrame(
        ord_enc.fit_transform(X_ordi),
        columns=X_ordi.columns,
        index=X_ordi.index
    )

    # -------- LABEL ENCODING --------
    X_label = X_train.iloc[:, [4, 6, 8]].copy()

    for col in X_label.columns:
        le = LabelEncoder()
        X_label[col] = le.fit_transform(X_label[col])

    # -------- ONE HOT ENCODING --------
    X_onehot = X_train.iloc[:, [1, 5, 7]]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    ohe_array = ohe.fit_transform(X_onehot)

    ohe_cols = ohe.get_feature_names_out(X_onehot.columns)

    X_onehot_enc = pd.DataFrame(
        ohe_array,
        columns=ohe_cols,
        index=X_onehot.index
    )

    # -------- COMBINE --------
    X_final = pd.concat([X_ordi_enc, X_label, X_onehot_enc], axis=1)

    print("\nFinal Encoded Data")
    print(X_final.head())

    # -------- L1 NORM --------
    l1_values = l1_norm_dataset(X_final)
    print("\nL1 Norm Values")
    print(l1_values[:5])

    # -------- L2 NORM --------
    l2_values = l2_norm_dataset(X_final)
    print("\nL2 Norm Values")
    print(l2_values[:5])


if __name__ == "__main__":
    main()