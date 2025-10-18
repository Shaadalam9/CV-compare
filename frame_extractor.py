# by md_shadab_alam@outlook.com
"""
Frame extraction from videos based on YOLO-detected objects (mapping-first, AV1-robust).

Flow:
1) Read mapping.csv row-by-row (city, country, videos).
2) For each video_id in that row:
   a) Find that video's CSV files in BBOX_DIR (pattern: {video_id}_{start}_{fps}.csv).
   b) Select frames meeting thresholds + continuous window.
   c) Locate the corresponding .mp4 in any configured video_dirs.
   d) Extract frames, saving as: {city}_{country}_{videoid}_{frame_number}.jpg
"""

import ast
import csv
import glob
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple
import cv2
import polars as pl
import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

# Toggle: if True, the script stops after the FIRST mapping row with a non-empty 'country'.
STOP_AFTER_FIRST_VALID_MAPPING_ROW = False


def find_frames_with_real_index(
    csv_path: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
    window: int = 10
) -> Tuple[str, int, pl.DataFrame]:
    """
    Identify valid frame indices from a YOLO detection CSV file based on detection thresholds and
    temporal stability over a rolling window of frames.

    The function reads YOLO detection CSV files that contain object detections with columns such as
    `frame-count` and `yolo-id`. It computes the number of persons, cars, and traffic lights per
    frame, checks if these counts meet given thresholds, and applies a rolling window to ensure
    stability (i.e., frames where these conditions hold true for a consecutive set of frames).

    Args:
        csv_path (str): Path to the YOLO detection CSV file.
        min_persons (int): Minimum required count of detected persons in a frame.
        min_cars (int): Minimum required count of detected cars in a frame.
        min_lights (int): Minimum required count of detected traffic lights in a frame.
        window (int, optional): Minimum number of consecutive frames that must satisfy all
            detection criteria to be considered "stable." Defaults to 10.

    Returns:
        Tuple[str, int, pl.DataFrame]:
            - video_id (str): Extracted from CSV filename.
            - fps (int): Frames per second, also extracted from filename.
            - valid_frames (pl.DataFrame): Filtered dataframe containing:
                * frame-count (frame index within the segment)
                * persons, cars, traffic_lights (object counts)
                * criteria_met (bool for threshold satisfaction)
                * stable_window (bool for rolling stability)
                * real-frame (absolute frame index in the full video)
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

    grouped = (
        df.group_by("frame-count")
        .agg([
            (pl.col("yolo-id") == 0).sum().alias("persons"),
            (pl.col("yolo-id") == 2).sum().alias("cars"),
            (pl.col("yolo-id") == 9).sum().alias("traffic_lights"),
        ])
        .sort("frame-count")
        .with_columns(
            ((pl.col("persons") >= min_persons)
             & (pl.col("cars") >= min_cars)
             & (pl.col("traffic_lights") >= min_lights)
            ).alias("criteria_met")
        )
        .with_columns(
            pl.col("criteria_met").rolling_min(window_size=window, min_samples=window).alias("stable_window")
        )
    )

    valid_frames = grouped.filter(pl.col("stable_window").eq(True))
    offset = start_time * fps
    valid_frames = valid_frames.with_columns((pl.col("frame-count") + offset).alias("real-frame"))

    return video_id, fps, valid_frames


def glob_csvs_for_video(bbox_dir: str, video_id: str) -> List[str]:
    """
    Find all YOLO detection CSV files associated with a particular video ID within a directory.

    Each YOLO detection CSV corresponds to a time segment of a video and follows the naming pattern:
    `{video_id}_{start_time}_{fps}.csv`.

    Args:
        bbox_dir (str): Path to the directory containing YOLO CSVs.
        video_id (str): The target video identifier.

    Returns:
        List[str]: Sorted list of full CSV file paths found for the given video.
                   Returns an empty list if no files match.
    """
    pattern = os.path.join(bbox_dir, f"{video_id}_*_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        logger.warning("No CSVs found for video_id='{}' in {}", video_id, bbox_dir)
    return paths


def select_frames_for_csvs(
    csv_paths: List[str],
    min_persons: int,
    min_cars: int,
    min_lights: int,
    max_frames: int,
    window: int
) -> List[int]:
    """
    Aggregate and select valid frame indices across multiple YOLO CSV segments for a given video.

    This function iterates over YOLO-generated CSV files corresponding to a single video, identifies
    stable frames using `find_frames_with_real_index`, and collects them based on spacing intervals
    and frame count limits. It ensures that selected frames are evenly spaced and meet all
    object-count thresholds.

    Args:
        csv_paths (List[str]): List of YOLO detection CSV file paths for one video.
        min_persons (int): Minimum number of detected persons.
        min_cars (int): Minimum number of detected cars.
        min_lights (int): Minimum number of detected traffic lights.
        max_frames (int): Maximum number of frames to select across all CSVs.
        window (int): Rolling window size for stable detection.

    Returns:
        List[int]: Sorted list of absolute frame numbers to extract from the video.
                   Returns an empty list if no frames meet the criteria.
    """
    if not csv_paths:
        return []

    collected: List[int] = []
    fps_for_spacing: Optional[int] = None

    for csv_path in csv_paths:
        _, fps, valid_frames_df = find_frames_with_real_index(csv_path, min_persons, min_cars, min_lights, window)
        if valid_frames_df.is_empty():
            logger.info("No valid frames in CSV: {}", csv_path)
            continue

        if fps_for_spacing is None:
            fps_for_spacing = fps

        step = (fps_for_spacing or 30) * common.get_configs("frame_interval")
        next_target = collected[-1] + step if collected else 0

        for row in valid_frames_df.iter_rows(named=True):
            rf = row["real-frame"]
            if not collected or rf >= next_target:
                collected.append(rf)
                next_target = rf + step
            if len(collected) >= max_frames:
                break

        if len(collected) >= max_frames:
            break

    logger.info("Selected {} frames for this video", len(collected))
    return collected


def parse_videos_list_field(videos_str: str) -> List[str]:
    """
    Parse and normalize the 'videos' column from mapping.csv into a list of clean video IDs.

    The field may contain Python list syntax or a simple comma-separated string.
    This helper ensures consistent parsing regardless of format.

    Args:
        videos_str (str): Raw string from the 'videos' field of mapping.csv.

    Returns:
        List[str]: List of normalized video IDs (without quotes or brackets).
                   Returns an empty list if parsing fails or field is empty.
    """
    if not videos_str:
        return []

    try:
        data = ast.literal_eval(videos_str)
        if isinstance(data, (list, tuple)):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    parts = [p.strip().strip("'").strip('"') for p in videos_str.strip("[]").split(",")]
    return [p for p in parts if p]


def build_time_interval_mapping(row: dict) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Construct a detailed mapping of videos to their time-of-day intervals (day/night segments).

    Used when a video has both day (0) and night (1) intervals. Extracts start and end frame
    information from mapping.csv for each segment.

    Args:
        row (dict): A row from mapping.csv as a dictionary, containing keys like 'videos',
            'start_time', 'end_time', and 'time_of_day'.

    Returns:
        Dict[str, List[Tuple[int, int, int]]]:
            Mapping of video_id -> list of (start_frame, end_frame, time_of_day),
            where `time_of_day` = 0 for day and 1 for night.
    """
    mapping: Dict[str, List[Tuple[int, int, int]]] = {}
    try:
        videos_list = parse_videos_list_field(row.get("videos", ""))
        time_field = ast.literal_eval(row.get("time_of_day", "[]"))
        start_field = ast.literal_eval(row.get("start_time", "[]"))
        end_field = ast.literal_eval(row.get("end_time", "[]"))

        for i, vid in enumerate(videos_list):
            try:
                tod_list = time_field[i]
                start_list = start_field[i]
                end_list = end_field[i]

                if not isinstance(tod_list, list):
                    tod_list = [tod_list]
                if not isinstance(start_list, list):
                    start_list = [start_list]
                if not isinstance(end_list, list):
                    end_list = [end_list]

                intervals = []
                for j, tod in enumerate(tod_list):
                    try:
                        s = start_list[j]
                        e = end_list[j]
                        intervals.append((s, e, int(tod)))
                    except Exception:
                        continue

                if intervals:
                    mapping[vid] = intervals
            except Exception:
                continue
    except Exception as e:
        logger.warning("Failed to build interval mapping: {}", e)
    return mapping


def get_video_mapping(mapping_csv_path) -> Dict[str, Tuple[str, str, object]]:
    """
    Load and parse the mapping.csv file to create a dictionary of video metadata.

    Each video_id maps to its corresponding (city, country, and time-of-day information).
    Supports both simple (single time-of-day) and multi-segment (both day and night) mappings.

    Args:
        mapping_csv_path (str): Path to mapping.csv.

    Returns:
        Dict[str, Tuple[str, str, object]]:
            video_id -> (city, country, tod_info)
            where `tod_info` can be:
                * int (0=day, 1=night)
                * list of intervals [(start, end, tod), ...] for multi-segment videos.
    """
    video_mapping: Dict[str, Tuple[str, str, object]] = {}

    try:
        with open(mapping_csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                city = row.get("city", "")
                country = row.get("country", "")
                videos_str = row.get("videos", "")
                videos_list = parse_videos_list_field(videos_str)

                time_field = row.get("time_of_day", "")
                time_of_day = 0
                nested = []

                try:
                    nested = ast.literal_eval(time_field)
                    flattened = [item for sublist in nested for item in sublist if isinstance(item, int)]
                    if 0 in flattened and 1 in flattened:
                        time_of_day = [0, 1]
                    elif 1 in flattened:
                        time_of_day = 1
                    else:
                        time_of_day = 0
                except Exception:
                    time_of_day = 0
                    nested = []

                if isinstance(time_of_day, list) and 0 in time_of_day and 1 in time_of_day:
                    interval_mapping = build_time_interval_mapping(row)
                    for vid, intervals in interval_mapping.items():
                        video_mapping[vid] = (city, country, intervals)
                    continue

                for i, vid in enumerate(videos_list):
                    tod = 0
                    try:
                        if i < len(nested):
                            if 1 in [int(x) for x in nested[i] if isinstance(x, int)]:
                                tod = 1
                    except Exception:
                        tod = 0
                    video_mapping[vid] = (city, country, tod)

        logger.info("Loaded video mapping for {} entries", len(video_mapping))
    except Exception as e:
        logger.error("Could not load video mapping from {}. Error: {}", mapping_csv_path, e)

    return video_mapping


def _ffprobe_codec(video_path: str) -> Optional[str]:
    """
    Retrieve the codec name of the first video stream using ffprobe.

    Args:
        video_path (str): Full path to the video file.

    Returns:
        Optional[str]: Codec name (e.g., 'h264', 'av1') if available, otherwise None.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nw=1:nk=1",
        video_path
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return out or None
    except Exception:
        return None


def _reencode_to_h264_sw(video_path: str) -> Optional[str]:
    """
    Re-encode a video to H.264 format using software decoding to ensure OpenCV compatibility.

    AV1 and certain hardware-encoded videos may not open properly with OpenCV.
    This function re-encodes them using FFmpeg (libx264 codec) and stores a temporary copy.

    Args:
        video_path (str): Full path to the input video file.

    Returns:
        Optional[str]: Path to the re-encoded H.264 video file, or None if the process fails.
    """
    base, _ = os.path.splitext(video_path)
    out_path = base + "_reencoded.mp4"
    if os.path.exists(out_path):
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "none",
        "-i", video_path,
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        "-map", "0:a?",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out_path if os.path.exists(out_path) else None
    except Exception as e:
        logger.error("FFmpeg software re-encode failed for {}: {}", video_path, e)
        return None


def safe_video_capture(video_path: str) -> Optional[cv2.VideoCapture]:
    """
    Safely open a video file for reading, automatically re-encoding if the codec is unsupported.

    The function first probes the codec; if AV1 or other unsupported formats are detected,
    it re-encodes the video to H.264 using `_reencode_to_h264_sw()` before opening it with OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Optional[cv2.VideoCapture]: Opened VideoCapture object if successful, otherwise None.
    """
    codec = _ffprobe_codec(video_path)
    if codec and codec.lower() in {"av1"}:
        reenc = _reencode_to_h264_sw(video_path)
        if reenc:
            cap = cv2.VideoCapture(reenc)
            if cap.isOpened():
                logger.info("Opened AV1 source via software reencode: {}", reenc)
                return cap

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap

    reenc = _reencode_to_h264_sw(video_path)
    if reenc:
        cap2 = cv2.VideoCapture(reenc)
        if cap2.isOpened():
            logger.info("Opened source via fallback reencode: {}", reenc)
            return cap2

    logger.error("Could not open video (original or reencoded): {}", video_path)
    return None


def save_frames_with_mapping(
    video_path: str,
    frame_numbers: List[int],
    save_dir: str,
    video_mapping: Dict[str, Tuple[str, str, object]]
) -> None:
    """
    Extract and save specific video frames according to the metadata mapping (city, country, time-of-day).

    Frames are saved using a structured filename format:
        `{city}_{country}_{videoid}_{frame_number}.jpg`

    If time-of-day information is provided (day/night), frames are saved in subdirectories named
    accordingly. Handles both single and multi-segment (day-night) videos.

    Args:
        video_path (str): Full path to the source video file.
        frame_numbers (List[int]): List of frame indices to extract.
        save_dir (str): Root directory where extracted frames are saved.
        video_mapping (Dict[str, Tuple[str, str, object]]): Mapping dictionary that links video_id
            to (city, country, and time-of-day or interval data).
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = safe_video_capture(video_path)
    if cap is None or not cap.isOpened():
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

    city, country, tod_info = video_mapping[video_id]
    multi_segment = isinstance(tod_info, list)

    if not multi_segment:
        time_label = "night" if tod_info == 1 else "day"
        subdir = os.path.join(save_dir, time_label)
        os.makedirs(subdir, exist_ok=True)

        for frame_no in frame_numbers:
            if frame_no >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, frame = cap.read()
            if success:
                out_name = f"{city}_{country}_{video_id}_{frame_no}.jpg"
                cv2.imwrite(os.path.join(subdir, out_name), frame)
        cap.release()
        logger.info("Saved {} frames for {} ({})", len(frame_numbers), video_id, time_label)
        return

    # Multi-segment (day/night intervals)
    for start, end, tod in tod_info:
        label = "night" if tod == 1 else "day"
        subdir = os.path.join(save_dir, label)
        os.makedirs(subdir, exist_ok=True)
        segment_frames = [f for f in frame_numbers if start <= f <= end]

        for frame_no in segment_frames:
            if frame_no >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, frame = cap.read()
            if success:
                out_name = f"{city}_{country}_{video_id}_{frame_no}.jpg"
                cv2.imwrite(os.path.join(subdir, out_name), frame)

        logger.info("Saved {} frames for {} ({}) segment", len(segment_frames), video_id, label)

    cap.release()


def main() -> None:
    """
   Orchestrates the complete **mapping-first frame extraction workflow**.

    This function serves as the top-level entry point for processing a batch
    of videos and extracting representative frames based on object detection data
    (from YOLO-generated bounding box CSVs). It relies on configuration values,
    mapping metadata, and threshold-based frame selection logic.

    **Workflow Steps:**
        1. Load configuration values from the common configuration utility.
           These include directory paths, frame thresholds, and operational limits.
        2. Load and parse the video-to-(city, country) mapping file (`mapping.csv`).
        3. For each valid row (where `country` is non-empty):
            - Parse the list of associated video IDs.
            - Locate all YOLO CSVs corresponding to each video.
            - Select valid frame numbers based on object thresholds
              (e.g., minimum people, cars, or lights detected).
            - Find the actual video file in the configured directories.
            - Extract and save the selected frames using the mapping metadata.
        4. Optionally stop after the first successfully processed mapping row
           if `STOP_AFTER_FIRST_VALID_MAPPING_ROW` is set to True.

    **Configuration Keys (Required):**
        - `BBOX_DIR` (str): Path to folder containing YOLO bounding box CSV files.
        - `video_dirs` (list[str]): Directories to search for the corresponding video files.
        - `SAVE_DIR` (str): Base directory for saving extracted frames.
        - `MIN_PERSONS`, `MIN_CARS`, `MIN_LIGHTS` (int): Detection thresholds.
        - `MAX_FRAMES` (int): Cap on frames extracted per video.
        - `CONF_WINDOW` (int): Confidence window for selection logic.
        - `mapping` (str): Path to the video mapping CSV file.

    **Error Handling:**
        - Logs detailed errors if configurations are missing or unreadable.
        - Skips any mapping rows lacking a valid `country` field.
        - Continues gracefully on missing videos or CSVs, logging each case.

    **Logging Behavior:**
        - Info logs summarize progress per city/country and per video.
        - Warnings are issued for incomplete or skipped data.
        - Errors include context (video ID or row) for traceability.

    **Returns:**
        None — This is a top-level driver function that performs all
        side effects (file I/O, CSV reading, frame extraction, logging)
        without returning any intermediate results.

    **Notes:**
        - Execution stops early if `STOP_AFTER_FIRST_VALID_MAPPING_ROW=True`.
        - Ensure that both YOLO CSVs and video files share consistent video IDs.
        - Designed for robust batch operation on large-scale datasets.
    """
    try:
        bbox_dir = common.get_configs("BBOX_DIR")
        video_dirs = common.get_configs("video_dirs")
        save_dir: str = common.get_configs("SAVE_DIR")
        min_persons: int = common.get_configs("MIN_PERSONS")
        min_cars: int = common.get_configs("MIN_CARS")
        min_lights: int = common.get_configs("MIN_LIGHTS")
        max_frames: int = common.get_configs("MAX_FRAMES")
        window: int = common.get_configs("CONF_WINDOW")
        mapping_csv_path = common.get_configs("mapping")
    except KeyError as e:
        logger.error("Missing required configuration key: {}", e)
        return
    except Exception as e:
        logger.error("Configuration loading failed. Error: {}", e)
        return

    video_mapping = get_video_mapping(mapping_csv_path)
    if not video_mapping:
        logger.error("Empty or unreadable mapping.csv; nothing to do.")
        return

    processed_any_row = False

    try:
        with open(mapping_csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                city = row.get("city", "")
                country = row.get("country", "")
                videos_str = row.get("videos", "")

                if not country:
                    logger.warning("Row for city='{}' has empty country; skipping.", city)
                    continue

                videos_list = parse_videos_list_field(videos_str)
                logger.info(
                    "Processing mapping row: city='{}', country='{}', videos={}",
                    city,
                    country,
                    len(videos_list),
                )

                for video_id in videos_list:
                    csv_paths = glob_csvs_for_video(bbox_dir, video_id)
                    if not csv_paths:
                        logger.warning("No CSVs for video_id='{}'; skipping.", video_id)
                        continue

                    frame_numbers = select_frames_for_csvs(
                        csv_paths, min_persons, min_cars, min_lights, max_frames, window
                    )
                    if not frame_numbers:
                        logger.info("No frames matched thresholds for video_id='{}'.", video_id)
                        continue

                    found_video: Optional[str] = None
                    for folder in video_dirs:
                        candidate = os.path.join(folder, f"{video_id}.mp4")
                        if os.path.exists(candidate):
                            found_video = candidate
                            break

                    if not found_video:
                        logger.error("Video file for ID '{}' not found in any directory.", video_id)
                        continue

                    save_frames_with_mapping(found_video, frame_numbers, save_dir, video_mapping)

                processed_any_row = True

                if STOP_AFTER_FIRST_VALID_MAPPING_ROW:
                    logger.info("STOP_AFTER_FIRST_VALID_MAPPING_ROW=True — stopping after this row.")
                    break

        if not processed_any_row:
            logger.warning("No mapping rows with a non-empty 'country' were processed.")
    except Exception as e:
        logger.error("Failed while iterating mapping.csv rows. Error: {}", e)


if __name__ == "__main__":
    main()
