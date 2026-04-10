import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load and Clean
data = pd.read_csv(r"C:\Users\My Hp\Desktop\energy.csv")
data.columns = data.columns.str.strip()
data = data.dropna(subset=['power_max', 'available_duration'])

# 2. Define Features and Target
X = data[['available_duration']]
y = data['power_max']

# 3. Train the Model
model = LinearRegression()
model.fit(X, y)

# 4. GET THE SLOPE AND INTERCEPT
slope = model.coef_[0]
intercept = model.intercept_

print("-" * 30)
print(f"📈 Slope (m): {slope:.4f}")
print(f"🏠 Intercept (b): {intercept:.4f}")
print("-" * 30)
print(f"Final Equation: Power = ({slope:.4f} * Duration) + {intercept:.4f}")