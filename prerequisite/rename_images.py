import os
import json
import shutil

# Paths
json_file = 'data/annotations.json'
downloaded_dir = 'data/downloaded_images'     # directory where flickr_640 images are saved
output_root = 'data/organized_images'         # destination root directory

# Load JSON
with open(json_file, 'r') as f:
    data = json.load(f)

noOfFails = 0

# Loop through images in JSON
for item in data['images']:
    flickr_url = item['flickr_url']
    target_path = item['file_name']  # e.g., "batch_1/000006.jpg"

    # Get filename from flickr_url
    flickr_filename = flickr_url.split('/')[-1]

    # Full source and target paths
    src = os.path.join(downloaded_dir, flickr_filename)
    dst = os.path.join(output_root, target_path)

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Copy and rename
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        noOfFails += 1
        print(f"Missing: {flickr_filename}")

print("Renaming and organizing complete.")
print(f"Number of missing files: {noOfFails}")
