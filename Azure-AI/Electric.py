mport pandas as pd
import numpy as np

# 1. Load the dataset
# Replace 'your_file_name.csv' with the actual name of your file
file_path = r"C:\Users\My Hp\Desktop\energy.csv" 
data = pd.read_csv(file_path, sep=',', na_values='?', low_memory=False)

# 2. Clean the column names (removes hidden spaces like ' appliance_category')
data.columns = data.columns.str.strip()

# 3. Handle missing values 
# This removes rows where data is missing (like the '?' seen in your previous error)
data = data.dropna()

# 4. Convert specific columns to numbers
# Based on your image, 'power_max' should be a numeric column
if 'power_max' in data.columns:
    data['power_max'] = pd.to_numeric(data['power_max'], errors='coerce')

# 5. Display the first 20 rows
print(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.\n")
print(data.head(20))