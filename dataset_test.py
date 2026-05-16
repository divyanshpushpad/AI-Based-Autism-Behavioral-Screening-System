import pandas as pd

# Load dataset
data = pd.read_csv('dataset/autism_dataset.csv')

# Show first 5 rows
print(data.head())

# Show dataset shape
print("\nDataset Shape:")
print(data.shape)

# Show column names
print("\nColumns:")
print(data.columns)

# Show missing values
print("\nMissing Values:")
print(data.isnull().sum())
