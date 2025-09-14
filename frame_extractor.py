"""Frame extraction utilities for YOLO-annotated videos.

This module reads YOLO CSV outputs, identifies frames that satisfy
object-count thresholds, and saves them as images.

Linting rules:
- Line length: ≤119 (flake8 E501)
- Docstrings follow Google style.
"""

import os
import glob
import re
from typing import List, Tuple

import cv2
import polars as pl

from common import get_configs
from custom_logger import CustomLogger

# Initialize a custom logger for this module
logger = CustomLogger(__name__)


def find_frames_with_real_index(
    csv_path: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
) -> Tuple[str, int, pl.DataFrame]:
    """Identify valid frames from a YOLO CSV based on object counts.

    Args:
        csv_path (str): Path to the YOLO CSV containing detection results.
        min_persons (int): Minimum required number of persons per frame.
        min_cars (int): Minimum required number of cars per frame.
        min_lights (int): Minimum required number of traffic lights per frame.

    Returns:
        Tuple[str, int, pl.DataFrame]:
            - video_id: Identifier parsed from the CSV filename.
            - fps: Frames per second for the video (parsed from the filename).
            - valid_frames: A Polars DataFrame containing frames meeting the thresholds
              with an added ``real-frame`` column.
    """
    filename = os.path.basename(csv_path)

    # Filename format expected: <video_id>_<start_time>_<fps>.csv
    match = re.match(r"(.+?)_(\d+)_(\d+)\.csv", filename)
    if not match:
        logger.warning("CSV filename %s does not match the expected pattern", filename)
        return "", 0, pl.DataFrame()

    video_id, start_time_str, fps_str = match.groups()
    start_time, fps = int(start_time_str), int(fps_str)

    # Read YOLO detection results
    df = pl.read_csv(csv_path)

    # Aggregate counts of each object per frame
    grouped = df.group_by("frame-count").agg(
        [
            (pl.col("yolo-id") == 0).sum().alias("persons"),         # YOLO ID 0 → person
            (pl.col("yolo-id") == 2).sum().alias("cars"),            # YOLO ID 2 → car
            (pl.col("yolo-id") == 9).sum().alias("traffic_lights"),  # YOLO ID 9 → traffic light
        ]
    )

    # Offset to compute real frame index (start_time * fps)
    offset: int = start_time * fps

    # Filter frames that satisfy thresholds and compute real-frame number
    valid_frames = (
        grouped
        .filter(
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
    video_ids: List[str],
    bbox_dir: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
    max_frames: int,
) -> List[Tuple[str, int]]:
    """Collect valid frames for all videos belonging to a specific city.

    Args:
        city (str): Name of the city (for logging only).
        video_ids (List[str]): List of video IDs to process.
        bbox_dir (str): Directory containing YOLO CSV annotations.
        min_persons (int): Minimum number of persons per frame.
        min_cars (int): Minimum number of cars per frame.
        min_lights (int): Minimum number of traffic lights per frame.
        max_frames (int): Maximum number of frames to extract overall.

    Returns:
        List[Tuple[str, int]]: A list of tuples ``(video_id, real_frame_index)``.
    """
    found_frames: List[Tuple[str, int]] = []
    logger.info("Processing city: %s", city)

    for vid in video_ids:
        pattern = os.path.join(bbox_dir, f"{vid}_*.csv")
        csv_paths = glob.glob(pattern)

        for csv_path in csv_paths:
            video_id, fps, valid_frames_df = find_frames_with_real_index(
                csv_path, min_persons, min_cars, min_lights
            )

            if valid_frames_df.is_empty():
                continue

            # Extract at most one frame every 10 minutes (600 s)
            step = fps * 600
            next_target = 0  # Next frame allowed based on step

            # Iterate over candidate frames
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
    """Save selected frames as JPEG images.

    Args:
        video_path (str): Path to the video file.
        frame_numbers (List[int]): List of frame indices to save.
        save_dir (str): Directory to store the extracted frames.

    Notes:
        - Frames that cannot be read will be skipped with a warning.
        - The directory will be created if it does not exist.
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
    """Entry point for frame extraction.

    Reads configuration values, selects frames for the configured city,
    and saves them as images.
    """
    # Load configuration values from the config module
    city = get_configs("CITY_NAME")
    video_ids = get_configs("VIDEO_IDS")
    bbox_dir = get_configs("BBOX_DIR")
    video_path = get_configs("VIDEO_PATH")
    save_dir = get_configs("SAVE_DIR")

    frames = select_frames_for_city(
        city=city,
        video_ids=video_ids,
        bbox_dir=bbox_dir,
        min_persons=get_configs("MIN_PERSONS"),
        min_cars=get_configs("MIN_CARS"),
        min_lights=get_configs("MIN_LIGHTS"),
        max_frames=get_configs("MAX_FRAMES"),
    )

    frame_numbers = [frame[1] for frame in frames]
    save_frames(video_path, frame_numbers, save_dir)


if __name__ == "__main__":
    main()
