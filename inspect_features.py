import numpy as np

# Load the saved features with allow_pickle=True
features = np.load("audiojack_features.npy", allow_pickle=True)

# Print basic details
print("Type of loaded data:", type(features))
print("Shape of extracted features:", features.shape if isinstance(features, np.ndarray) else "Not a NumPy array")
print("First 5 feature vectors:\n", features[:5])
