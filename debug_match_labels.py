import numpy as np

# Load original labels and filenames
all_labels = np.load("/Users/jasmineborse/Desktop/Project_2/IAD/audiojack_labels.npy", allow_pickle=True)
all_filenames = np.load("/Users/jasmineborse/Desktop/Project_2/IAD/audiojack_filenames.npy", allow_pickle=True)

# Load fused filenames
fused_filenames = np.load("/Users/jasmineborse/Desktop/Project_2/IAD/audiojack_fused_filenames.npy", allow_pickle=True)

# Create a mapping from filename to label
filename_to_label = dict(zip(all_filenames, all_labels))

# Debug: Check if original labels contain anomalies (1)
unique_labels, counts = np.unique(all_labels, return_counts=True)
print("✅ Unique Labels in Original Data:", unique_labels)
print("🔢 Counts:", counts)

# Debug: Check how filenames are paired in fused data
print("\n🔍 First 5 Fused Filenames:", fused_filenames[:5])

# Debug: Try to retrieve labels for the first 5 fused pairs
for pair in fused_filenames[:5]:
    if isinstance(pair, (list, tuple, np.ndarray)) and len(pair) == 2:
        label1 = filename_to_label.get(pair[0], "Not Found")
        label2 = filename_to_label.get(pair[1], "Not Found")
        fused_label = max(label1, label2)
    else:
        fused_label = filename_to_label.get(pair, "Not Found")

    print(f"📂 Pair: {pair} → Labels: {label1}, {label2} → Fused Label: {fused_label}")
