import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_data():
    data = pd.read_csv("breast-cancer.csv", header=None, na_values='?')
    X = data.iloc[:, :9]
    Y = data.iloc[:, 9]
    return X,Y

def main():
    X, Y = load_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 25)

    imputer = SimpleImputer(strategy='most_frequent')
    X_train = pd.DataFrame(imputer.fit_transform(X_train))
    X_test = pd.DataFrame(imputer.fit_transform(X_test))

# X,Y = data_processing(X,Y)

    LE = LabelEncoder()
    Y_train = LE.fit_transform(Y_train)
    Y_test = LE.transform(Y_test)

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value= -1), [0,2,3,4,6,8]),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1,5,7])
        ]
    )

    #Fit transform
    X_trained_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    #Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_trained_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    #Model training
    model = LogisticRegression()
    model.fit(X_train_scaled, Y_train)

    #evaluation

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, y_pred)

    print(f"Model Results")
    print(f"Accuracy: {accuracy}")



if __name__ == "__main__":
    main()