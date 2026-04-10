import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load Data
file_path = r"C:\Users\My Hp\Desktop\energy.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
data = data.dropna(subset=['power_max', 'available_duration'])

# 2. Create the Target (Using Median ensures a mix of 0s and 1s)
median_val = data['power_max'].median()
data['High_Usage'] = (data['power_max'] > median_val).astype(int)

# 3. Define X and y
X = data[['available_duration']]
y = data['High_Usage']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. SCALE THE DATA (Crucial to avoid getting only 0s)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Final Output
y_pred = model.predict(X_test_scaled)
print("-" * 35)
print(f" Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("-" * 35)
print("Detailed Predictions (Categories):")
print(y_pred)