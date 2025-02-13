import numpy as np

# Load fused features & filenames
fused_features = np.load("audiojack_fused.npy", allow_pickle=True)
fused_filenames = np.load("audiojack_fused_filenames.npy", allow_pickle=True)

# Load all labels & filenames
all_labels = np.load("audiojack_labels.npy", allow_pickle=True)
all_filenames = np.load("audiojack_filenames.npy", allow_pickle=True)

# Convert 'OK' and 'NG' labels to numerical values if needed
if all_labels.dtype.type is np.str_:
    label_mapping = {"OK": 0, "NG": 1}  # Modify if NG has multiple classes
    all_labels = np.array([label_mapping.get(label, 1) for label in all_labels])

# Create a mapping of filenames to labels
filename_to_label = dict(zip(all_filenames.tolist(), all_labels.tolist()))

# Assign labels based on fused filenames
matched_labels = []
missing_labels = 0

for fused_pair in fused_filenames:
    files = fused_pair.split("|")  # Ensure filenames are split correctly
    labels = [filename_to_label.get(f, None) for f in files]  # Get labels

    if None in labels:  # If any label is missing, warn the user
        print(f"🚨 Missing label for: {files} → Found labels: {labels}")
        missing_labels += 1

    fused_label = max(filter(None, labels), default=0)  # Assign max label (default 0)
    matched_labels.append(fused_label)

matched_labels = np.array(matched_labels)

# Save the matched labels
np.save("audiojack_fused_labels.npy", matched_labels)
print("✅ Matched labels saved!")

# Debugging: Check label distribution
unique_labels, counts = np.unique(matched_labels, return_counts=True)
print("✅ Unique Labels in Fused Data:", unique_labels)
print("🔢 Counts:", counts)

# Ensure shape consistency
if fused_features.shape[0] != matched_labels.shape[0]:
    print(f"❌ Mismatch: Features {fused_features.shape[0]} ≠ Labels {matched_labels.shape[0]}")
else:
    print("✅ Shapes match correctly!")

# Final check for missing labels
if missing_labels > 0:
    print(f"⚠️ Warning: {missing_labels} missing labels detected!")
