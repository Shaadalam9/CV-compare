"""
Frame extraction from videos based on YOLO-detected objects.

This script reads YOLO CSV outputs, filters frames by minimum object counts,
and saves the selected frames as images. It supports multiple video folders
and does not require city-specific configuration.
"""

import glob
import os
import re
from typing import List, Tuple

import cv2
import polars as pl

from common import get_configs, root_dir
from custom_logger import CustomLogger
import subprocess
logger = CustomLogger(__name__)


def find_frames_with_real_index(
    csv_path: str, min_persons: int, min_cars: int, min_lights: int,
    window: int = 10,
) -> Tuple[str, int, pl.DataFrame]:
    """
    1. Minimum object counts (persons, cars, traffic lights)
    2. Confidence consistency over a rolling window (default 10 frames)

    Only frames where all criteria are satisfied simultaneously are returned.
    """
    filename = os.path.basename(csv_path)
    match = re.match(r"(.+?)_(\d+)_(\d+)\.csv", filename)
    if not match:
        logger.warning("Skipped CSV due to unexpected filename format: {}", filename)
        return "", 0, pl.DataFrame()

    video_id, start_time_str, fps_str = match.groups()
    start_time, fps = int(start_time_str), int(fps_str)

    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        logger.error("Unable to read CSV file {}. Error: {}", csv_path, e)
        return video_id, fps, pl.DataFrame()

    grouped = df.group_by("frame-count").agg(
        [
            (pl.col("yolo-id") == 0).sum().alias("persons"),
            (pl.col("yolo-id") == 2).sum().alias("cars"),
            (pl.col("yolo-id") == 9).sum().alias("traffic_lights"),
        ]
    ).sort("frame-count")
    # Step 1: mark frames that satisfy all criteria
    grouped = grouped.with_columns(
    (
        (pl.col("persons") >= min_persons) &
        (pl.col("cars") >= min_cars) &
        (pl.col("traffic_lights") >= min_lights)
    ).alias("criteria_met")
)
    
    
    # Apply rolling min over the window to ensure criteria are met continuously
    grouped = grouped.with_columns(
        pl.col("criteria_met").rolling_min(window_size=window, min_periods=window).alias("stable_window")
    )

    # Keep only frames where the criteria were continuously satisfied
    valid_frames = grouped.filter(pl.col("stable_window") == True)

    # Add real-frame offset
    offset = start_time * fps
    valid_frames = valid_frames.with_columns((pl.col("frame-count") + offset).alias("real-frame"))

    return video_id, fps, valid_frames


def select_frames(
    bbox_dir: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
    max_frames: int,
    window: int,
) -> List[Tuple[str, int]]:
    """
    Collect valid frames from all CSVs in a given directory.
    """
    found_frames: List[Tuple[str, int]] = []
    logger.info("Scanning CSV files in directory: {}", bbox_dir)

    csv_paths = glob.glob(os.path.join(bbox_dir, "*.csv"))
    if not csv_paths:
        logger.warning("No CSV files were found in: {}", bbox_dir)
        return []

    for csv_path in csv_paths:
        video_id, fps, valid_frames_df = find_frames_with_real_index(
            csv_path, min_persons, min_cars, min_lights,window,
        )
        if valid_frames_df.is_empty():
            logger.info("No valid frames found in CSV: {}", csv_path)
            continue

        step = fps * 600  # spacing: every 10 minutes
        next_target = 0

        for row in valid_frames_df.iter_rows(named=True):
            if row["real-frame"] >= next_target:
                found_frames.append((video_id, row["real-frame"]))
                next_target = row["real-frame"] + step
            if len(found_frames) >= max_frames:
                break
        if len(found_frames) >= max_frames:
            break

    logger.info("Total frames selected: {}", len(found_frames))
    return found_frames


def save_frames(
    video_path: str, frame_numbers: List[int], save_dir: str
) -> None:
    """
    Save selected frames from a video as images.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Failed to open video file: {}", video_path)
        return

    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(
                save_dir,
                f"{os.path.basename(video_path)}_"
                f"frame_{frame_num}.jpg",
            )
            cv2.imwrite(out_path, frame)
            logger.info("Extracted and saved frame {} -> {}", frame_num, out_path)
        else:
            logger.warning("Failed to read frame {} from video {}", frame_num, video_path)

    cap.release()


import csv


def get_video_mapping(mapping_csv_path: str) -> dict:
    """
    Build a dictionary mapping video_id -> (city, country)
    """
    video_mapping = {}
    try:
        with open(mapping_csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                city = row['city']
                country = row['country']

                videos_str = row['videos']
                videos_list = videos_str.strip("[]").replace("'", "").split(",")
                for vid in videos_list:
                    vid = vid.strip()
                    if vid:
                        video_mapping[vid] = (city, country)
        logger.info("Loaded video mapping for {} entries", len(video_mapping))
    except Exception as e:
        logger.error("Could not load video mapping from {}. Error: {}", mapping_csv_path, e)
    return video_mapping

def safe_video_capture(video_path: str) -> cv2.VideoCapture:
    """Try to open video; return None if OpenCV cannot decode it."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(
            "Failed to open video file: {}. It may be an unsupported codec (e.g., AV1).",
            video_path,
        )
        temp_path = os.path.splitext(video_path)[0] + "_reencoded.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vcodec", "libx264", "-acodec", "aac",
            temp_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            cap = cv2.VideoCapture(temp_path)
            if cap.isOpened():
                logger.info("Fallback succeeded — using reencoded file: {}", temp_path)
                return cap
            else:
                logger.error("Fallback reencode also failed for {}", video_path)
        except Exception as e:
            logger.error("FFmpeg reencode failed for {}: {}", video_path, e)
        return None
    return cap


def save_frames_with_mapping(
    video_path: str, frame_numbers: List[int], save_dir: str, video_mapping: dict
) -> None:
    """
    Save frames with filenames: {city}_{country}_{videoid}_{frame_number}.jpg
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = safe_video_capture(video_path)

    if not cap.isOpened():
        logger.error("Failed to open video file: {}", video_path)
        return

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_id not in video_mapping:
        logger.error(
            "Video ID '{}' not found in mapping.csv. Frames will be skipped for this video.",
            video_id,
        )
        cap.release()
        return
    city, country = video_mapping[video_id]

    for frame_num in frame_numbers:
        if frame_num >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            filename = f"{city}_{country}_{video_id}_{frame_num}.jpg"
            out_path = os.path.join(save_dir, filename)
            cv2.imwrite(out_path, frame)
            logger.info("Saved frame {} with mapping -> {}", frame_num, out_path)
        else:
            logger.warning("Failed to read frame {} from {}", frame_num, video_path)

    cap.release()

def main() -> None:
    """
    Full workflow:
    - Reads config for directories and thresholds
    - Generates video mapping from internal CSV
    - Selects frames based on YOLO CSV outputs
    - Saves frames with {city}_{country}_{videoid}_{frame_number} naming
    """
    try:
        bbox_dir = get_configs("BBOX_DIR")
        video_dirs = get_configs("video_dirs")
        save_dir = get_configs("SAVE_DIR")

        min_persons = get_configs("MIN_PERSONS")
        min_cars = get_configs("MIN_CARS")
        min_lights = get_configs("MIN_LIGHTS")
        max_frames = get_configs("MAX_FRAMES")
        window = get_configs("CONF_WINDOW")      # e.g., 10–15

        mapping_csv_path = os.path.join(root_dir, "mapping.csv")
    except KeyError as e:
        logger.error("Missing required configuration key: {}", e)
        return
    except Exception as e:
        logger.error("Configuration loading failed. Error: {}", e)
        return

    video_mapping = get_video_mapping(mapping_csv_path)

    frames = select_frames(bbox_dir, min_persons, min_cars, min_lights, max_frames,window)
    if not frames:
        logger.warning("No frames matched the current selection criteria.")
        return

    frame_numbers = [f[1] for f in frames]

    # Instead of iterating over video_paths, iterate over mapping.csv entries
    for video_id, (city, country) in video_mapping.items():
        # Try to locate the actual video file inside video_dirs
        found_video = None
        for folder in video_dirs:
            candidate = os.path.join(folder, f"{video_id}.mp4")
            if os.path.exists(candidate):
                found_video = candidate
                break

        if not found_video:
            logger.error("Video file for ID '{}' not found in any directory.", video_id)
            continue

        save_frames_with_mapping(found_video, frame_numbers, save_dir, video_mapping)



if __name__ == "__main__":
    main()




