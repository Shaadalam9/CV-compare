# frame_extractor.py

import os
import glob
import re
import polars as pl
import cv2

# --------- CONFIG ---------
MAPPING_CSV = "mapping.csv"         # Path to your mapping.csv
BBOX_DIR = "data/bbox"              # Path to folder with CSVs
OUTPUT_DIR = "frames"               # Folder where frames will be saved
MIN_PERSONS = 3
MIN_CARS = 5
MIN_LIGHTS = 1
MAX_FRAMES = 3                      

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------- FUNCTIONS ---------
def find_frames_with_real_index(csv_path, min_persons=3, min_cars=5, min_lights=1):
    filename = os.path.basename(csv_path)
    m = re.match(r"(.+?)_(\d+)_(\d+)\.csv", filename)
    if not m:
        return None, None, pl.DataFrame()
    video_id, start_time, fps = m.groups()
    start_time, fps = int(start_time), int(fps)
    
    df = pl.read_csv(csv_path)
    
    grouped = df.group_by("frame-count").agg([
        (pl.col("yolo-id") == 0).sum().alias("persons"),
        (pl.col("yolo-id") == 2).sum().alias("cars"),
        (pl.col("yolo-id") == 9).sum().alias("traffic_lights")
    ])
    
    valid_frames = grouped.filter(
        (pl.col("persons") >= min_persons) &
        (pl.col("cars") >= min_cars) &
        (pl.col("traffic_lights") >= min_lights)
    ).with_columns(
        (pl.col("frame-count") + start_time * fps).alias("real-frame")
    ).sort("frame-count")
    
    return video_id, fps, valid_frames


def select_frames_for_city(mapping_df, city_name, bbox_dir, min_persons=3, min_cars=5, min_lights=1, max_frames=3):
    row = mapping_df.filter(pl.col("city") == city_name)
    if row.is_empty():
        print(f"❌ No entry found for city {city_name}")
        return []
    
    video_ids = row[0, "videos_with_csv"]
    found_frames = []
    
    print(f"🏙️ Processing city: {city_name}")
    
    for vid in video_ids:
        pattern = os.path.join(bbox_dir, f"{vid}_*.csv")
        csv_paths = glob.glob(pattern)
        
        for csv_path in csv_paths:
            video_id, fps, valid_frames = find_frames_with_real_index(csv_path, min_persons, min_cars, min_lights)
            if valid_frames.is_empty():
                continue
            
            step = fps * 600  # 10 min apart
            next_target = 0
            
            for row_f in valid_frames.iter_rows(named=True):
                if row_f["real-frame"] >= next_target:
                    found_frames.append(row_f["real-frame"])
                    next_target = row_f["real-frame"] + step
                if len(found_frames) >= max_frames:
                    break
            if len(found_frames) >= max_frames:
                break
    
    print(f"✅ Collected {len(found_frames)} frames for {city_name}")
    return found_frames


def save_frames(video_path, frame_numbers, output_dir=OUTPUT_DIR):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video:", video_path)
        return
    
    frame_numbers = sorted(frame_numbers)
    saved_count = 0
    for i, frame_num in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(output_dir, f"{i}_{frame_num}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
    cap.release()
    print(f"✅ Saved {saved_count} frames to {output_dir}")


# --------- MAIN ---------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from video based on YOLO CSVs")
    parser.add_argument("--city", type=str, required=True, help="Name of the city")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    # Load mapping
    mapping = pl.read_csv(MAPPING_CSV)

    # Make sure 'videos_with_csv' column exists
    if "videos_with_csv" not in mapping.columns:
        # Convert the 'videos' string to list
        mapping = mapping.with_columns(
            pl.col("videos").apply(lambda s: [v.strip("[]'\" ") for v in s.split(",")] if s else []).alias("videos_with_csv")
        )

    # Select frames
    frames = select_frames_for_city(mapping, args.city, BBOX_DIR, MIN_PERSONS, MIN_CARS, MIN_LIGHTS, MAX_FRAMES)

    # Save frames
    save_frames(args.video, frames)
