import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load non-fused features & labels
features = np.load("audiojack_features_fixed.npy", allow_pickle=True)
labels = np.load("audiojack_labels_fixed.npy", allow_pickle=True)

# Extract only the feature arrays
features = np.array([np.array(f["features"], dtype=np.float32) for f in features])

# Convert labels to NumPy array (ensuring correct dtype)
labels = np.array(labels, dtype=np.float32)

# Convert to PyTorch tensors
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Reshape for compatibility

# Debugging: Check final shapes
print(f"✅ Non-Fused Features Shape: {X.shape}, Labels Shape: {y.shape}")

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyClassifier(input_dim=X.shape[1]).to(device)
criterion = nn.BCELoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
torch.save(model.state_dict(), "anomaly_classifier_nonfused.pth")
print("✅ Non-Fused Model training complete! Model saved as anomaly_classifier_nonfused.pth")
