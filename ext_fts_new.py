import numpy as np

# Load the already extracted features
features = np.load("audiojack_features_fixed.npy", allow_pickle=True)

# Load filenames and labels separately
filenames = np.load("audiojack_filenames.npy", allow_pickle=True)
labels = np.load("audiojack_labels.npy", allow_pickle=True)

# Ensure lengths match
if len(features) != len(filenames) or len(features) != len(labels):
    print(f"❌ Mismatch: Features ({len(features)}), Filenames ({len(filenames)}), Labels ({len(labels)})")
else:
    # Convert to structured format
    feature_data = [{"filename": filename, "features": feature, "label": label}
                    for filename, feature, label in zip(filenames, features, labels)]

    # Save in the correct format
    np.save("audiojack_features_fixed.npy", np.array(feature_data, dtype=object))
    print("✅ Fixed features saved in correct format!")
