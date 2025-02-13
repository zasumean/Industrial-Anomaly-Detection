import os
import json
import re
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Load JSON metadata
json_path = "/Users/jasmineborse/Desktop/Project_2/IAD/realiad_jsons/jsons/audiojack.json"
with open(json_path, "r") as f:
    metadata = json.load(f)

# Extract camera view from filename
def get_view_from_filename(filename):
    match = re.search(r"_C(\d)_", filename)
    return f"C{match.group(1)}" if match else None

# Model mapping for views
view_model_mapping = {
    "C1": "resnet50",
    "C2": "efficientnet_b0",
    "C3": "vit_base_patch16_224",
    "C4": "convnext_tiny",
    "C5": "densenet121"
}

# Load models (removing classification layers)
models_dict = {
    "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    "vit_base_patch16_224": models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
    "convnext_tiny": models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
    "densenet121": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
}

for key in models_dict:
    model = models_dict[key]
    if key == "vit_base_patch16_224":
        model.eval()  # ViT remains unchanged
    else:
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    
    model.eval()
    models_dict[key] = model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    view = get_view_from_filename(image_path)
    model_name = view_model_mapping.get(view, "resnet50")  # Default to ResNet50
    model = models_dict[model_name]

    with torch.no_grad():
        features = model(image)
        features = features.flatten().numpy()  # Flatten to 1D

    return features

# Process images and store features
def process_images(folder_path, save_features, save_filenames, save_labels):
    feature_list = []
    filename_list = []
    label_list = []

    for category in ["OK", "NG"]:
        category_path = os.path.join(folder_path, category)

        if not os.path.exists(category_path):
            print(f"⚠️ Skipping {category} - Path does not exist: {category_path}")
            continue  # Skip if folder does not exist

        # Handle OK category
        if category == "OK":
            subfolders = [os.path.join(category_path, sub) for sub in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, sub))]
        else:
            # Handle NG category
            subfolders = []
            for anomaly_folder in os.listdir(category_path):
                anomaly_path = os.path.join(category_path, anomaly_folder)
                if os.path.isdir(anomaly_path):
                    for sub in os.listdir(anomaly_path):
                        sub_path = os.path.join(anomaly_path, sub)
                        if os.path.isdir(sub_path):
                            subfolders.append(sub_path)

        # Process all images
        for subdir_path in subfolders:
            for file in os.listdir(subdir_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(subdir_path, file)
                    features = extract_features(image_path)

                    # ✅ Store full relative path instead of just the filename
                    relative_filename = os.path.relpath(image_path, folder_path)

                    # ✅ Match labels correctly
                    label = metadata.get(file, {}).get("anomaly", 1) if category == "NG" else 0

                    feature_list.append(features)
                    filename_list.append(relative_filename)  # ✅ Store correct filenames
                    label_list.append(label)

    # Convert feature list to a NumPy array
    max_length = max(f.shape[0] for f in feature_list)  # Find max feature length
    features_padded = np.array([np.pad(f, (0, max_length - f.shape[0])) for f in feature_list])

    # ✅ Save correct filenames
    np.save(save_features, features_padded)
    np.save(save_filenames, np.array(filename_list, dtype=object))
    np.save(save_labels, np.array(label_list, dtype=np.int32))

    print(f"✅ Features saved to {save_features}")
    print(f"✅ Filenames saved to {save_filenames}")
    print(f"✅ Labels saved to {save_labels}")

# Run feature extraction
process_images(
    "/Users/jasmineborse/Desktop/Project_2/IAD/audiojack",
    "audiojack_features_fixed.npy",
    "audiojack_filenames_fixed.npy",
    "audiojack_labels_fixed.npy"
)

