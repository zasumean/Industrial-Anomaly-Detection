import torch
from torchvision import transforms
from PIL import Image
import pickle

# Load the split dataset
split_data_path = "split_data.pkl"
with open(split_data_path, "rb") as f:
    train_data, val_data, test_data = pickle.load(f)

print(f"Loaded {len(train_data)} training samples")

import os
from PIL import Image

# Debug: Print first image path
sample_image_path = train_data[0][0]  # First image path from train set
print("Sample Image Path:", sample_image_path)

# Check if the file actually exists
if not os.path.exists(sample_image_path):
    print("⚠️ ERROR: File does not exist!")
else:
    image = Image.open(sample_image_path).convert("RGB")  # Open image
    print("Image loaded successfully!")


# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1,1] range
])

# Test on one image
sample_image_path = train_data[0][0]  # Get an image path from train set
image = Image.open(sample_image_path).convert("RGB")  # Open image

# Apply transformation
tensor_image = transform(image)

print("Image shape after transformation:", tensor_image.shape)  # (C, H, W)
print("Pixel value range:", tensor_image.min().item(), "to", tensor_image.max().item())


