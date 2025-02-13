import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=256):
        super(LightweightCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, features1, features2):
        Q = self.query_proj(features1)  
        K = self.key_proj(features2)
        V = self.value_proj(features2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attended_features = torch.matmul(attn_weights, V)

        return self.out_proj(attended_features)


def preprocess_features(features):
    processed_features = []

    for f in features:
        if f.ndimension() == 1:  
            f = f.unsqueeze(0)
        elif f.ndimension() == 4:  
            f = f.view(f.shape[0], -1)
        processed_features.append(f)

    max_dim = max(f.shape[1] for f in processed_features)
    processed_features = [F.pad(f, (0, max_dim - f.shape[1])) for f in processed_features]

    return processed_features


def load_features(feature_file):
    data = np.load(feature_file, allow_pickle=True)
    
    if isinstance(data, np.ndarray) and isinstance(data[0], dict):
        feature_list = [torch.tensor(f["features"], dtype=torch.float32) for f in data]
        filenames = [f["filename"] for f in data]  
        labels = [f.get("label", "OK") for f in data]  
        
        feature_list = preprocess_features(feature_list)  
        
        print("✅ Successfully extracted and processed feature tensors.")
        print("Standardized Feature Shapes:", [f.shape for f in feature_list])

        return feature_list, filenames, labels
    else:
        print("⚠️ Unexpected data format. Ensure the file contains a list of dictionaries.")
        return None, None, None


def fuse_features(feature_file, save_file):
    features, filenames, labels = load_features(feature_file)
    if features is None or len(features) < 2:
        print("❌ Not enough feature sets for fusion.")
        return
    
    input_dim = features[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model = LightweightCrossAttention(input_dim=input_dim).to(device)

    fused_data = []
    
    for i in range(0, len(features) - 1, 2):  # Fuse pairs of features
        fused = fusion_model(features[i].to(device), features[i+1].to(device))
        fused = fused.cpu().detach().numpy()

        fused_data.append({
            "filename": f"{filenames[i]}|{filenames[i+1]}",  # Store as a single string
            "features": fused.flatten(),  # Ensure flattened format
            "label": (labels[i], labels[i+1])  
        })

    np.save(save_file, np.array(fused_data, dtype=object))  # Save as array of dicts
    
    print(f"✅ Fused features saved to {save_file}")


if __name__ == "__main__":
    fuse_features(
        "audiojack_features_fixed.npy",
        "audiojack_fused.npy"
    )
