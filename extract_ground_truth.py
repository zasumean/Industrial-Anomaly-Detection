import json
import numpy as np

# Define a mapping for NG subclasses
defect_mapping = {
    "AK": 1,  # Pit
    "BX": 2,  # Deformation
    "CH": 3,  # Abrasion
    "HS": 4,  # Scratch
    "PS": 5,  # Damage
    "QS": 6,  # Missing parts
    "YW": 7,  # Foreign objects
    "ZW": 8   # Unknown/Other
}

def extract_ground_truth(json_file, output_labels, output_filenames):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    labels = []
    filenames = []

    # Process train and test data
    for split in ["train", "test"]:
        for entry in data[split]:
            filenames.append(entry["image_path"])
            
            if entry["anomaly_class"] == "OK":
                labels.append(0)  # Normal
            else:
                # Assign correct defect type using mapping
                labels.append(defect_mapping.get(entry["anomaly_class"], 8))  # Default to 8 if unknown

    # Save labels and filenames
    np.save(output_labels, np.array(labels))
    np.save(output_filenames, np.array(filenames))

    print(f"✅ Saved labels to {output_labels}")
    print(f"✅ Saved filenames to {output_filenames}")

# Run the function
extract_ground_truth("realiad_jsons/jsons/audiojack.json", "audiojack_labels.npy", "audiojack_filenames.npy")
