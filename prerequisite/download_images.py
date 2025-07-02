import csv
import os
import requests
from tqdm import tqdm

CSV_FILE = "data/all_image_urls.csv"
SAVE_DIR = "data/downloaded_images"

os.makedirs(SAVE_DIR, exist_ok=True)

with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    for row in tqdm(reader):
        original_url = row[1].strip()  # second column
        filename = original_url.split("/")[-1]
        filepath = os.path.join(SAVE_DIR, filename)

        if not os.path.exists(filepath):
            try:
                r = requests.get(original_url, timeout=10)
                with open(filepath, "wb") as out:
                    out.write(r.content)
            except Exception as e:
                print(f"Failed to download {original_url}: {e}")