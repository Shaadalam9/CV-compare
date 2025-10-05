"""
Keep at most 3 frames per country based on saved frame filenames.

Filename format: {city}_{country}_{videoid}_{frame_number}.jpg
Example: New Delhi_India_Af0SUwah1bQ_40577.jpg
"""

import os
from collections import defaultdict
import re
from common import get_configs  

# Directory containing already saved frames
FRAME_DIR =  get_configs("SAVE_DIR")  # <--- change this


# Maximum frames per country to keep
MAX_FRAMES_PER_COUNTRY = 3

# Regex to extract city and country from filename
FILENAME_PATTERN = re.compile(r"^(.*?)_(.*?)_.*\.jpg$")

# Map country -> list of frame file paths
frames_by_country = defaultdict(list)

# Scan directory
for fname in os.listdir(FRAME_DIR):
    if not fname.lower().endswith(".jpg"):
        continue
    match = FILENAME_PATTERN.match(fname)
    if not match:
        print(f"Skipping invalid filename: {fname}")
        continue
    city, country = match.groups()
    frames_by_country[country].append(os.path.join(FRAME_DIR, fname))

# Delete frames exceeding the max per country
for country, files in frames_by_country.items():
    # Sort by filename (you can sort differently if you want oldest/newest)
    files.sort()
    if len(files) > MAX_FRAMES_PER_COUNTRY:
        to_delete = files[MAX_FRAMES_PER_COUNTRY:]
        for fpath in to_delete:
            os.remove(fpath)
        print(f"{len(to_delete)} frames deleted for country {country}. Kept {MAX_FRAMES_PER_COUNTRY} frames.")
    else:
        print(f"Country {country} has {len(files)} frames. No deletion needed.")
