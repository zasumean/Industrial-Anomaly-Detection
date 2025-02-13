import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load test features & labels for both fused and non-fused models
X_fused_test = np.load("audiojack_fused_test.npy", allow_pickle=True)
y_fused_test = np.load("audiojack_fused_test_labels.npy", allow_pickle=True)

X_nonfused_test = np.load("audiojack_nonfused_test.npy", allow_pickle=True)
y_nonfused_test = np.load("audiojack_nonfused_test_labels.npy", allow_pickle=True)

# Convert to tensors
X_fused_test = torch.tensor(np.array([np.array(f["features"], dtype=np.float32) for f in X_fused_test]), dtype=torch.float32)
y_fused_test = torch.tensor(y_fused_test, dtype=torch.float32)

X_nonfused_test = torch.tensor(np.array([np.array(f["features"], dtype=np.float32) for f in X_nonfused_test]), dtype=torch.float32)
y_nonfused_test = torch.tensor(y_nonfused_test, dtype=torch.float32)

# Define the model class
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

# Load both models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fused Model
model_fused = AnomalyClassifier(input_dim=X_fused_test.shape[1]).to(device)
model_fused.load_state_dict(torch.load("anomaly_classifier.pth", map_location=device))
model_fused.eval()

# Non-Fused Model
model_nonfused = AnomalyClassifier(input_dim=X_nonfused_test.shape[1]).to(device)
model_nonfused.load_state_dict(torch.load("anomaly_classifier_nonfused.pth", map_location=device))
model_nonfused.eval()

# Get predictions for both models
with torch.no_grad():
    X_fused_test, X_nonfused_test = X_fused_test.to(device), X_nonfused_test.to(device)
    
    y_pred_fused = model_fused(X_fused_test).cpu().numpy().flatten()
    y_pred_nonfused = model_nonfused(X_nonfused_test).cpu().numpy().flatten()

# Compute ROC-AUC scores
roc_auc_fused = roc_auc_score(y_fused_test.numpy(), y_pred_fused)
roc_auc_nonfused = roc_auc_score(y_nonfused_test.numpy(), y_pred_nonfused)

print(f"✅ Fused Model ROC-AUC Score: {roc_auc_fused:.4f}")
print(f"✅ Non-Fused Model ROC-AUC Score: {roc_auc_nonfused:.4f}")

# Plot ROC Curves
fpr_fused, tpr_fused, _ = roc_curve(y_fused_test.numpy(), y_pred_fused)
fpr_nonfused, tpr_nonfused, _ = roc_curve(y_nonfused_test.numpy(), y_pred_nonfused)

plt.figure(figsize=(8, 6))
plt.plot(fpr_fused, tpr_fused, color="blue", label=f"Fused Model (AUC = {roc_auc_fused:.4f})")
plt.plot(fpr_nonfused, tpr_nonfused, color="red", label=f"Non-Fused Model (AUC = {roc_auc_nonfused:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: Fused vs Non-Fused Model")
plt.legend()
plt.grid()
plt.show()
