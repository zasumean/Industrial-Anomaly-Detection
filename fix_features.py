import numpy as np

# Load the incorrectly saved features
features = np.load("audiojack_features.npy", allow_pickle=True)

# Wrap features in dictionaries with placeholder filenames
fixed_features = [{"filename": f"image_{i}.jpg", "features": feat} for i, feat in enumerate(features)]

# Save the corrected features
np.save("audiojack_features_fixed.npy", np.array(fixed_features, dtype=object))

print("✅ Fixed features saved as 'audiojack_features_fixed.npy'.")
