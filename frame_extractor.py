import os
import cv2
import glob
import polars as pl
from custom_logger import get_logger
from default.config import CONFIG

# Initialize logger
logger = get_logger("FrameExtractor")

# Ensure results directory exists
RESULTS_DIR = CONFIG["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_mapping():
    """Load mapping CSV using Polars."""
    mapping_path = CONFIG["mapping_csv"]
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping CSV not found at {mapping_path}")
        return pl.DataFrame()
    mapping = pl.read_csv(mapping_path)
    return mapping

def clean_video_list(series):
    """Convert string list to Python list."""
    cleaned = []
    for s in series:
        if s:
            cleaned.append([v.strip("[]").strip().strip("'").strip('"') for v in s.split(",")])
        else:
            cleaned.append([])
    return pl.Series(cleaned)

def filter_available_videos(series, csv_dir):
    """Keep only videos that have corresponding CSVs."""
    all_csvs = glob.glob(os.path.join(csv_dir, "*.csv"))
    csv_video_ids = [os.path.basename(f).split("_")[0] for f in all_csvs]
    filtered = []
    for vids in series:
        filtered.append([v for v in vids if v in csv_video_ids])
    return pl.Series(filtered)

def find_frames_with_real_index(csv_path, min_persons, min_cars, min_lights):
    """Return frames that satisfy the object count thresholds."""
    filename = os.path.basename(csv_path)
    m = glob.fnmatch.fnmatch(filename, "*.csv")
    if not m:
        return None, None, pl.DataFrame()

    # Extract video_id, start_time, fps from CSV name
    try:
        parts = filename.replace(".csv", "").split("_")
        video_id, start_time, fps = parts[0], int(parts[1]), int(parts[2])
    except Exception:
        logger.warning(f"Filename {filename} not in expected format")
        return None, None, pl.DataFrame()

    df = pl.read_csv(csv_path)
    grouped = df.groupby("frame-count").agg([
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

def save_frames(video_file, frame_numbers, output_dir=RESULTS_DIR):
    """Extract and save frames from video."""
    if not os.path.exists(video_file):
        logger.error(f"Video file not found: {video_file}")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_file}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video {video_file} opened, total frames: {total_frames}")

    for frame_no in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Frame {frame_no} could not be read")
            continue
        out_path = os.path.join(output_dir, f"{os.path.basename(video_file)}_frame_{frame_no}.jpg")
        cv2.imwrite(out_path, frame)
        logger.info(f"Saved frame {frame_no} -> {out_path}")

    cap.release()

def main():
    """Main function to extract frames from all videos listed in mapping."""
    mapping = load_mapping()
    if mapping.is_empty():
        logger.error("Mapping CSV is empty or missing. Exiting...")
        return

    # Clean and filter video lists
    mapping = mapping.with_columns(clean_video_list(mapping["videos"]).alias("video_list"))
    mapping = mapping.with_columns(filter_available_videos(mapping["video_list"], CONFIG["bbox_dir"]).alias("videos_with_csv"))

    # Iterate over cities
    for row in mapping.iter_rows(named=True):
        city = row["city"]
        videos = row["videos_with_csv"]
        logger.info(f"Processing city: {city}")

        for vid in videos:
            csv_pattern = os.path.join(CONFIG["bbox_dir"], f"{vid}_*.csv")
            csv_files = glob.glob(csv_pattern)
            for csv_file in csv_files:
                video_id, fps, valid_frames = find_frames_with_real_index(
                    csv_file,
                    CONFIG["min_persons"],
                    CONFIG["min_cars"],
                    CONFIG["min_lights"]
                )
                if valid_frames.is_empty():
                    continue

                frame_nums = valid_frames["real-frame"].to_list()[:CONFIG["max_frames"]]
                video_path = os.path.join(CONFIG["video_dir"], f"{vid}.mp4")
                save_frames(video_path, frame_nums)

if __name__ == "__main__":
    main()
