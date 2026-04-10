import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# 1. Load and automatically fix column names
file_path = r"C:\Users\My Hp\Desktop\energy.csv"
df = pd.read_csv(file_path, sep=None, engine='python')

# THIS IS THE FIX: Strip spaces and force everything to lowercase
df.columns = df.columns.str.strip().str.lower()

# Print this to verify what the columns are called now
print("Cleaned columns:", df.columns.tolist())

# 2. Encoding (Use 'appliance' now because we forced it to lowercase)
le = LabelEncoder()
df['appliance_encoded'] = le.fit_transform(df['appliance'])

# Normalize for the network
df['duration_norm'] = df['duration'] / df['duration'].max()
df['power_norm'] = df['power_max'] / df['power_max'].max()

# 3. Define Q-Network
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

# 4. Run Model
model = EnergyQNetwork(3, 3)
first_row = df.iloc[0]
state = torch.FloatTensor([first_row['duration_norm'], first_row['appliance_encoded'], first_row['power_norm']])

print("State Tensor:", state)
print("Q-Values:", model(state))