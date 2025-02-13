import numpy as np

fused_features = np.load("audiojack_fused.npy", allow_pickle=True)
print("✅ Loaded fused features successfully!")
print("Fused features shape:", fused_features.shape)
print("Example entry:", fused_features[0])  # Inspect the first fused feature
