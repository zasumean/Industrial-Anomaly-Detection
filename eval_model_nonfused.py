import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test features & labels
X_test = np.load("audiojack_features.npy", allow_pickle=True)
y_test = np.load("audiojack_labels.npy", allow_pickle=True)

# Convert to NumPy array (ensure float dtype)
X_test = np.array([np.array(f, dtype=np.float32) for f in X_test])
y_test = np.array(y_test, dtype=np.float32)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Reshape for binary classification

# Define the model (same as in training)
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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyClassifier(input_dim=X_test.shape[1]).to(device)
model.load_state_dict(torch.load("anomaly_classifier_nonfused.pth", map_location=device))
model.eval()

# Run inference
with torch.no_grad():
    X_test = X_test.to(device)
    outputs = model(X_test).cpu().numpy()
    preds = (outputs >= 0.5).astype(int)  # Convert to binary predictions

# Convert tensors to NumPy for metrics
y_test_np = y_test.numpy()

# Compute metrics
accuracy = accuracy_score(y_test_np, preds)
precision = precision_score(y_test_np, preds)
recall = recall_score(y_test_np, preds)
f1 = f1_score(y_test_np, preds)

# Display results
print(f"✅ Non-Fused Model Evaluation:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
