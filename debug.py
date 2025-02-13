import numpy as np

# Check if files exist
test_features_path = "audiojack_fused_test.npy"
test_labels_path = "audiojack_fused_test_labels.npy"

try:
    test_features = np.load(test_features_path, allow_pickle=True)
    test_labels = np.load(test_labels_path, allow_pickle=True)

    print(f"✅ Loaded Test Features Shape: {test_features.shape}")
    print(f"✅ Loaded Test Labels Shape: {test_labels.shape}")

    # Check if they are actually empty
    if len(test_features) == 0:
        print("❌ Test features array is empty! Check your dataset split.")

    if len(test_labels) == 0:
        print("❌ Test labels array is empty! Check your dataset split.")

except Exception as e:
    print(f"❌ Error loading test data: {e}")
