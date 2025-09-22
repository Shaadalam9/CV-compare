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

from common import get_configs
from custom_logger import CustomLogger

logger = CustomLogger(__name__)


def find_frames_with_real_index(
    csv_path: str, min_persons: int, min_cars: int, min_lights: int
) -> Tuple[str, int, pl.DataFrame]:
    """
    Reads a YOLO CSV file and returns valid frame numbers
    based on minimum object counts.

    Args:
        csv_path: Path to the YOLO CSV file.
        min_persons: Minimum number of persons in the frame.
        min_cars: Minimum number of cars in the frame.
        min_lights: Minimum number of traffic lights in the frame.

    Returns:
        video_id: ID of the video extracted from the CSV filename.
        fps: Frames per second of the video.
        valid_frames: Polars DataFrame of frames meeting the
                      object count criteria.
    """
    filename = os.path.basename(csv_path)
    match = re.match(r"(.+?)_(\d+)_(\d+)\.csv", filename)
    if not match:
        logger.warning(
            "Skipping CSV with unexpected filename format: %s",
            filename,
        )
        return "", 0, pl.DataFrame()

    video_id, start_time_str, fps_str = match.groups()
    start_time, fps = int(start_time_str), int(fps_str)

    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        logger.error("Failed to read CSV %s: %s", csv_path, e)
        return video_id, fps, pl.DataFrame()

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


def select_frames(
    bbox_dir: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
    max_frames: int,
) -> List[Tuple[str, int]]:
    """
    Collect valid frames from all CSVs in a given directory.

    Args:
        bbox_dir: Directory containing YOLO CSV outputs.
        min_persons: Minimum number of persons per frame.
        min_cars: Minimum number of cars per frame.
        min_lights: Minimum number of traffic lights per frame.
        max_frames: Maximum number of frames to select.

    Returns:
        A list of tuples (video_id, frame_number) for selected frames.
    """
    found_frames: List[Tuple[str, int]] = []
    logger.info("Processing CSV files in %s", bbox_dir)

    csv_paths = glob.glob(os.path.join(bbox_dir, "*.csv"))
    if not csv_paths:
        logger.warning("No CSV files found in directory: %s", bbox_dir)
        return []

    for csv_path in csv_paths:
        video_id, fps, valid_frames_df = find_frames_with_real_index(
            csv_path, min_persons, min_cars, min_lights
        )
        if valid_frames_df.is_empty():
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

    logger.info("Collected %d frames in total", len(found_frames))
    return found_frames


def save_frames(
    video_path: str, frame_numbers: List[int], save_dir: str
) -> None:
    """
    Save selected frames from a video as images.

    Args:
        video_path: Path to the video file.
        frame_numbers: List of frame indices to save.
        save_dir: Directory to save output images.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
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
            logger.info(
                "Saved frame %d to %s",
                frame_num,
                out_path,
            )
        else:
            logger.warning(
                "Could not read frame %d from video %s",
                frame_num,
                video_path,
            )

    cap.release()


def main() -> None:
    """
    Main workflow for frame extraction.

    Reads configuration, collects all videos, selects frames from CSVs,
    and saves them to the output directory.
    """
    try:
        bbox_dir = get_configs("BBOX_DIR")
        video_dirs = get_configs("video_dirs")
        save_dir = get_configs("SAVE_DIR")

        min_persons = get_configs("MIN_PERSONS")
        min_cars = get_configs("MIN_CARS")
        min_lights = get_configs("MIN_LIGHTS")
        max_frames = get_configs("MAX_FRAMES")

    except KeyError as e:
        logger.error("Missing configuration key: %s", e)
        return
    except Exception as e:
        logger.error("Error reading configuration: %s", e)
        return

    video_paths = []
    for folder in video_dirs:
        folder_videos = glob.glob(os.path.join(folder, "*.mp4"))
        video_paths.extend(folder_videos)

    if not video_paths:
        logger.warning(
            "No video files found in directories: %s",
            video_dirs,
        )
        return

    frames = select_frames(
        bbox_dir, min_persons, min_cars, min_lights, max_frames
    )
    if not frames:
        logger.warning("No frames selected based on current criteria.")
        return

    frame_numbers = [f[1] for f in frames]

    for video_path in video_paths:
        save_frames(video_path, frame_numbers, save_dir)


if __name__ == "__main__":
    main()
