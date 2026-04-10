import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load and Clean
file_path = r"C:\Users\My Hp\Desktop\energy.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
data = data.dropna(subset=['power_max', 'available_duration'])

# 2. Select Features for PCA
# PCA needs multiple columns to "compress" them
X = data[['power_max', 'available_duration']]

# 3. SCALE THE DATA (Mandatory for PCA)
# PCA is based on distance, so all numbers must be on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA
# n_components=1 means we are squashing 2 columns into 1 main column
pca = PCA(n_components=1)
pca_result = pca.fit_transform(X_scaled)

# 5. Output Results
print("-" * 35)
print(f"✅ PCA Explained Variance Ratio: {pca.explained_variance_ratio_[0]:.4f}")
print("-" * 35)
print("First 5 compressed PCA values:")
print(pca_result)