import pandas as pd
import numpy as np
def main():

    data = pd.read_csv("data.csv")
    X = data.drop(["diagnosis", "id", "Unnamed: 32"], axis=1)
    y = data["diagnosis"]
    X = X.values
    x_new=(X - np.min(X))/(np.max(X)) - (np.min(X))
    print(x_new)
    print()
    print()
    print("---------------------------standardization--------------------------------")
    x_std=(X - np.mean(X))/(np.std(X))
    print(x_std)
if __name__ == "__main__":
    main()