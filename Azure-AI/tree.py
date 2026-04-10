import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
file_path = r"C:\Users\My Hp\Desktop\energy.csv"
data = pd.read_csv(file_path)

# 2. Clean data
data.columns = data.columns.str.strip()
data = data.dropna(subset=['power_max', 'available_duration'])

# 3. Create Classification Target (High vs Low)
# Decision Trees are best for categorizing data
avg_power = data['power_max'].mean()
data['target'] = (data['power_max'] > avg_power).astype(int)

# 4. Define Features (X) and Target (y)
X = data[['available_duration']]
y = data['target']

# 5. Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and Train the Decision Tree
# 'max_depth' prevents the tree from getting too complex (overfitting)
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# 7. Output Results
y_pred = model.predict(X_test)
print("-" * 30)
print(f"✅ Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("-" * 30)
print("Sample Predictions:, {y_pred}")