import numpy as np
import pickle

# Load split data
split_data_path = "split_data.pkl"
with open(split_data_path, "rb") as f:
    train_data, val_data, test_data = pickle.load(f)

# Extract filenames from the split data
train_filenames = set(f[0].split("/")[-1] for f in train_data)
val_filenames = set(f[0].split("/")[-1] for f in val_data)
test_filenames = set(f[0].split("/")[-1] for f in test_data)

# Load fused features, filenames, and labels
fused_features = np.load("audiojack_fused.npy", allow_pickle=True)
fused_filenames = np.load("audiojack_fused_filenames.npy", allow_pickle=True)
fused_labels = np.load("audiojack_fused_labels.npy", allow_pickle=True)

# Initialize lists for train, val, and test sets
train_fused, val_fused, test_fused = [], [], []
train_labels, val_labels, test_labels = [], [], []

skipped = 0  # Count of skipped pairs

for i, fused_pair in enumerate(fused_filenames):
    f1, f2 = fused_pair.split("|")  # Extract the two filenames
    
    if f1 in train_filenames and f2 in train_filenames:
        train_fused.append(fused_features[i])
        train_labels.append(fused_labels[i])
    elif f1 in val_filenames and f2 in val_filenames:
        val_fused.append(fused_features[i])
        val_labels.append(fused_labels[i])
    elif f1 in test_filenames and f2 in test_filenames:
        test_fused.append(fused_features[i])
        test_labels.append(fused_labels[i])
    else:
        skipped += 1  # Count discarded pairs

# Convert lists to numpy arrays
train_fused = np.array(train_fused)
val_fused = np.array(val_fused)
test_fused = np.array(test_fused)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# Save the final split fused features
np.save("audiojack_fused_train.npy", train_fused)
np.save("audiojack_fused_val.npy", val_fused)
np.save("audiojack_fused_test.npy", test_fused)

np.save("audiojack_fused_train_labels.npy", train_labels)
np.save("audiojack_fused_val_labels.npy", val_labels)
np.save("audiojack_fused_test_labels.npy", test_labels)

# Print dataset sizes and skipped count
print(f"✅ Train Fused Samples: {train_fused.shape[0]}")
print(f"✅ Validation Fused Samples: {val_fused.shape[0]}")
print(f"✅ Test Fused Samples: {test_fused.shape[0]}")
print(f"⚠️ Discarded {skipped} fused pairs due to split mismatch.")
