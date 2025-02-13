import os
import json
import pprint

# Define the correct path to your extracted JSON files
json_folder = "/Users/jasmineborse/Desktop/Project_2/IAD/realiad_jsons/jsons"  # Replace with the actual path

json_path = os.path.join(json_folder, "audiojack.json")  # Replace with an actual file name

with open(json_path, "r") as f:
    data = json.load(f)


pprint.pprint(data, depth=1)

# Print the keys inside "meta"
print("Meta keys:", list(data["meta"].keys()))

# Print a small sample from "train"
print("First training entry:", json.dumps(data["train"][:1], indent=4))

# Print a small sample from "test"
print("First test entry:", json.dumps(data["test"][:1], indent=4))

#Convert image paths to full paths
base_image_path = "/Users/jasmineborse/Desktop/Project_2/IAD/audiojack"  # Adjust this if your images are elsewhere

# Example: Get full path of first training image
train_img_full_path = os.path.join(base_image_path, data["train"][0]["image_path"])
print("Full image path:", train_img_full_path)

#Check if anomalies have mask
anomaly_samples = [entry for entry in data["test"] if entry["mask_path"] is not None]

if anomaly_samples:
    print("Example of an anomaly with a mask:", json.dumps(anomaly_samples[:1], indent=4))
else:
    print("No mask files found in test data.")
