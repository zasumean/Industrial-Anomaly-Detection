import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
test_features = np.load("audiojack_fused.npy", allow_pickle=True)
test_labels = np.load("audiojack_fused_labels.npy", allow_pickle=True)

# Ensure correct feature extraction
if isinstance(test_features[0], dict) and "features" in test_features[0]:
    test_features = np.array([np.array(item["features"], dtype=np.float32) for item in test_features])
else:
    test_features = np.array([np.array(f, dtype=np.float32) for f in test_features])

# Convert labels to NumPy array
test_labels = np.array(test_labels, dtype=np.float32).reshape(-1, 1)

# Convert to PyTorch tensors
X_test = torch.tensor(test_features, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.float32)

# Define model architecture (must match the trained model)
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

# Load model & weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyClassifier(input_dim=X_test.shape[1]).to(device)
model.load_state_dict(torch.load("anomaly_classifier.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Make predictions
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary values

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
