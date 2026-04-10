import pandas as pd
from sklearn.model_selection import train_test_split
# IMPORTANT: This line defines the Random Forest
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load and Clean
file_path = r"C:\Users\My Hp\Desktop\energy.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
data = data.dropna(subset=['power_max', 'available_duration'])

# 2. Setup Target (High vs Low Power)
avg_power = data['power_max'].mean()
data['target'] = (data['power_max'] > avg_power).astype(int)

# 3. Define X and y
X = data[['available_duration']]
y = data['target']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest
# We use 200 trees to make it different from the single Decision Tree
model = RandomForestClassifier(n_estimators=200, random_state=7) 
model.fit(X_train, y_train)

# 6. Output ALL Predictions
y_pred = model.predict(X_test)

# Force terminal to show every value without dots
np.set_printoptions(threshold=np.inf)

print("-" * 35)
print(f" Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("-" * 35)
print(f"All Predictions: {y_pred}")