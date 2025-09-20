"""
Frame extraction from videos based on YOLO-detected objects.

This script reads YOLO CSV outputs, filters frames by minimum object counts,
and saves the selected frames as images.
"""

import glob
import os
import re
from typing import List, Tuple

import cv2
import polars as pl

from common import get_configs
from custom_logger import CustomLogger

logger = CustomLogger(__name__)


def find_frames_with_real_index(
    csv_path: str, min_persons: int, min_cars: int, min_lights: int
) -> Tuple[str, int, pl.DataFrame]:
    """
    Read YOLO CSV and return valid frame numbers based on min object counts.
    Only process CSVs that match the expected pattern.
    """
    filename = os.path.basename(csv_path)
    match = re.match(r"(.+?)_(\d+)_(\d+)\.csv", filename)
    if not match:
        # Skip CSVs that don't match the pattern
        return "", 0, pl.DataFrame()

    video_id, start_time_str, fps_str = match.groups()
    start_time, fps = int(start_time_str), int(fps_str)

    df = pl.read_csv(csv_path)

    grouped = df.group_by("frame-count").agg(
        [
            (pl.col("yolo-id") == 0).sum().alias("persons"),
            (pl.col("yolo-id") == 2).sum().alias("cars"),
            (pl.col("yolo-id") == 9).sum().alias("traffic_lights"),
        ]
    )

    offset = start_time * fps

    valid_frames = (
        grouped.filter(
            (pl.col("persons") >= min_persons)
            & (pl.col("cars") >= min_cars)
            & (pl.col("traffic_lights") >= min_lights)
        )
        .with_columns((pl.col("frame-count") + offset).alias("real-frame"))
        .sort("frame-count")
    )

    return video_id, fps, valid_frames


def select_frames_for_city(
    city: str,
    bbox_dir: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
    max_frames: int,
) -> List[Tuple[str, int]]:
    """
    Extract valid frames from all videos in the bbox directory.

    Args:
        city: Name of the city.
        bbox_dir: Directory containing YOLO CSV outputs.
        min_persons: Minimum persons per frame.
        min_cars: Minimum cars per frame.
        min_lights: Minimum traffic lights per frame.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of tuples (video_id, frame_number) for valid frames.
    """
    found_frames: List[Tuple[str, int]] = []
    logger.info("Processing city: %s", city)

    csv_paths = glob.glob(os.path.join(bbox_dir, "*.csv"))

    for csv_path in csv_paths:
        video_id, fps, valid_frames_df = find_frames_with_real_index(
            csv_path, min_persons, min_cars, min_lights
        )

        if valid_frames_df.is_empty():
            continue

        step = fps * 600  # every 10 minutes
        next_target = 0

        # Select frames spaced by 'step'
        for row in valid_frames_df.iter_rows(named=True):
            if row["real-frame"] >= next_target:
                found_frames.append((video_id, row["real-frame"]))
                next_target = row["real-frame"] + step
            if len(found_frames) >= max_frames:
                break
        if len(found_frames) >= max_frames:
            break

    logger.info("Collected %d frames for %s", len(found_frames), city)
    return found_frames


def save_frames(video_path: str, frame_numbers: List[int], save_dir: str) -> None:
    """
    Save selected frames as images.

    Args:
        video_path: Path to the video file.
        frame_numbers: List of frame indices to save.
        save_dir: Directory to save images.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Cannot open video %s", video_path)
        return

    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(save_dir, f"frame_{frame_num}.jpg")
            cv2.imwrite(out_path, frame)
            logger.info("Saved frame %d to %s", frame_num, out_path)
        else:
            logger.warning("Could not read frame %d", frame_num)

    cap.release()


def main() -> None:
    """
    Main workflow to extract frames based on city and video configuration.
    Supports multiple video folders.
    """
    city = get_configs("CITY_NAME")
    bbox_dir = get_configs("BBOX_DIR")
    video_dirs = get_configs("VIDEO_DIRS")
    save_dir = get_configs("SAVE_DIR")

    # Collect all videos from all folders
    video_paths = []
    for folder in video_dirs:
        folder_videos = glob.glob(os.path.join(folder, "*.mp4"))
        video_paths.extend(folder_videos)

    frames = select_frames_for_city(
        city,
        bbox_dir,
        get_configs("MIN_PERSONS"),
        get_configs("MIN_CARS"),
        get_configs("MIN_LIGHTS"),
        get_configs("MAX_FRAMES"),
    )

    frame_numbers = [f[1] for f in frames]

    # Save frames for all videos
    for video_path in video_paths:
        save_frames(video_path, frame_numbers, save_dir)


if __name__ == "__main__":
    main()
