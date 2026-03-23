import pandas as pd

# Step 1 — Load your simulated dataset
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

# print("Original Dataset:\n")
print(data)


# Step 2 — Function to partition dataset
def partition_dataset(df, threshold):
    left = df[df['BP'] <= threshold]
    right = df[df['BP'] > threshold]

    print(f"Partition for Threshold t = {threshold}")

    print("\nLeft Dataset (BP <= t):")
    print(left)

    print("\nRight Dataset (BP > t):")
    print(right)


# Step 3 — Apply thresholds
partition_dataset(data, 80)
partition_dataset(data, 78)
partition_dataset(data, 82)