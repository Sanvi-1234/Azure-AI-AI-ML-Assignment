import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
file_path =  r"C:\Users\My Hp\Desktop\energy.csv"
df = pd.read_csv(file_path, sep=None, engine='python')

# 2. AUTOMATIC COLUMN FIXING
# This removes spaces and forces everything to lowercase
df.columns = df.columns.str.strip().str.lower()

# IMPORTANT: See exactly what columns pandas found
print("Verified Columns in your file:", df.columns.tolist())

# 3. DYNAMIC COLUMN SELECTION
# This checks if 'appliance' exists, if not, it tries to find the closest match
col_name = 'appliance' if 'appliance' in df.columns else None
if col_name is None:
    # Try to find any column containing the word 'appliance'
    matches = [c for c in df.columns if 'appliance' in c]
    if matches:
        col_name = matches[0]
    else:
        raise KeyError("Could not find any column named 'appliance'. Check your CSV file!")

# 4. PREPROCESSING
le = LabelEncoder()
df['appliance_encoded'] = le.fit_transform(df[col_name])

# Normalize values for the network
df['duration_norm'] = df['duration'] / df['duration'].max()
df['power_norm'] = df['power_max'] / df['power_max'].max()

# 5. Q-NETWORK ARCHITECTURE
class EnergyQNetwork(nn.Module):
    def _init_(self, state_dim, action_dim):
        super(EnergyQNetwork, self)._init_()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)

# 6. INITIALIZE AND TEST
model = EnergyQNetwork(state_dim=3, action_dim=3)
first_row = df.iloc[0]
state = torch.FloatTensor([first_row['duration_norm'], first_row['appliance_encoded'], first_row['power_norm']])

print("\n✅ Success!")
print(f"State Tensor: {state}")
print(f"Predicted Q-values: {model(state).detach().numpy()}")