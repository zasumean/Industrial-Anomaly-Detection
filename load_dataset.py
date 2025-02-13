import os

json_folder = "/Users/jasmineborse/Desktop/Project_2/IAD/realiad_jsons/realiad_jsons"  # Replace with the actual path

# List all JSON files
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

print("Found JSON files:")
print(json_files)
