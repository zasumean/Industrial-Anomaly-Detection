import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load fused features & labels
fused_data = np.load("audiojack_fused.npy", allow_pickle=True)
fused_labels = np.load("audiojack_fused_labels.npy", allow_pickle=True)

# Ensure proper extraction of features
if isinstance(fused_data[0], dict) and "features" in fused_data[0]:
    fused_features = np.array([np.array(item["features"], dtype=np.float32) for item in fused_data])
else:
    fused_features = np.array([np.array(f, dtype=np.float32) for f in fused_data])

# Convert labels to NumPy array (ensuring correct dtype)
fused_labels = np.array(fused_labels, dtype=np.float32).reshape(-1, 1)  # Reshape for BCELoss

# Debug: Check shapes and types
print(f"✅ Fused Features Shape: {fused_features.shape}, Labels Shape: {fused_labels.shape}")

# Convert to PyTorch tensors
X = torch.tensor(fused_features, dtype=torch.float32)
y = torch.tensor(fused_labels, dtype=torch.float32)

# Define a simple classification model
class AnomalyClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Create DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyClassifier(input_dim=X.shape[1]).to(device)
criterion = nn.BCELoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), "anomaly_classifier.pth")
print("✅ Model training complete! Model saved as anomaly_classifier.pth")
