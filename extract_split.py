import os
import json
import random
from sklearn.model_selection import train_test_split
import pickle

# Define dataset paths
json_folder = "/Users/jasmineborse/Desktop/Project_2/IAD/realiad_jsons/jsons"  # JSON metadata folder
image_root = "/Users/jasmineborse/Desktop/Project_2/IAD/audiojack"  # Root directory of images

# Load a JSON file (example: audiojack.json)
json_path = os.path.join(json_folder, "audiojack.json")  # Change object name as needed
with open(json_path, "r") as f:
    data = json.load(f)

# Collect all image paths and labels
dataset = []
for entry in data["train"] + data["test"]:  # Combine train & test data
    full_image_path = os.path.join(image_root, entry["image_path"])

    label = entry["anomaly_class"]  # "OK" or an anomaly type
    dataset.append((full_image_path, label))

# Split into train (70%), val (15%), test (15%)
train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42, stratify=[label for _, label in dataset])
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=[label for _, label in temp_data])

# Print dataset sizes
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Sample output
print("Example train sample:", train_data[0])



# Save the split data
split_data_path = "split_data.pkl"
with open(split_data_path, "wb") as f:
    pickle.dump((train_data, val_data, test_data), f)

print(f"Saved split data to {split_data_path}")
