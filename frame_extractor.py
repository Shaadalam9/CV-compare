"""
Frame extraction from videos based on YOLO-detected objects (mapping-first, AV1-robust).

Flow:
1) Read mapping.csv row-by-row (city, country, videos).
2) For each video_id in that row:
   a) Find that video's CSV files in BBOX_DIR (pattern: {video_id}_{start}_{fps}.csv).
   b) Select frames meeting thresholds + continuous window.
   c) Locate the corresponding .mp4 in any configured video_dirs.
   d) Extract frames, saving as: {city}_{country}_{videoid}_{frame_number}.jpg

Improvements:
- Starts from mapping.csv as requested.
- Preserves linkage: frames from a video's CSVs are only applied to that same video.
- Robust AV1 handling: pre-detect codec with ffprobe; if AV1 (or open fails), re-encode once
  to H.264/AAC in software (-hwaccel none) and open the reencoded file.
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
from common import get_configs, root_dir
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

# Toggle: if True, the script stops after the FIRST mapping row with a non-empty 'country'.
STOP_AFTER_FIRST_VALID_MAPPING_ROW = False


def find_frames_with_real_index(
    csv_path: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
    window: int = 10,
) -> Tuple[str, int, pl.DataFrame]:
    """
    From a single YOLO CSV file:
      - Parse (video_id, start_time, fps) from filename: {video_id}_{start_time}_{fps}.csv
      - Count objects per frame (COCO ids: person=0, car=2, traffic light=9)
      - Keep frames that meet thresholds for a continuous 'window' of frames
      - Convert clip-relative 'frame-count' to video-absolute 'real-frame' using offset=start_time*fps
    Returns: (video_id, fps, valid_frames_df)
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
        .agg(
            [
                (pl.col("yolo-id") == 0).sum().alias("persons"),
                (pl.col("yolo-id") == 2).sum().alias("cars"),
                (pl.col("yolo-id") == 9).sum().alias("traffic_lights"),
            ]
        )
        .sort("frame-count")
        .with_columns(
            (
                (pl.col("persons") >= min_persons)
                & (pl.col("cars") >= min_cars)
                & (pl.col("traffic_lights") >= min_lights)
            ).alias("criteria_met")
        )
        .with_columns(
            pl.col("criteria_met")
            .rolling_min(window_size=window, min_samples=window)
            .alias("stable_window")
        )
    )

    valid_frames = grouped.filter(pl.col("stable_window") == True)

    offset = start_time * fps
    valid_frames = valid_frames.with_columns(
        (pl.col("frame-count") + offset).alias("real-frame")
    )

    return video_id, fps, valid_frames


def glob_csvs_for_video(bbox_dir: str, video_id: str) -> List[str]:
    """
    Return all CSVs belonging to a given video id.
    Expected pattern: {video_id}_{start_time}_{fps}.csv
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
    window: int,
) -> List[int]:
    """
    Collect valid *real-frame indices* for a single video by scanning its CSVs.
    Returns a list of frame numbers (absolute 'real-frame' in the video timeline),
    spaced ~10 minutes apart (using the fps from the first usable CSV).
    """
    if not csv_paths:
        return []

    collected: List[int] = []
    fps_for_spacing: Optional[int] = None

    for csv_path in csv_paths:
        _, fps, valid_frames_df = find_frames_with_real_index(
            csv_path, min_persons, min_cars, min_lights, window
        )
        if valid_frames_df.is_empty():
            logger.info("No valid frames in CSV: {}", csv_path)
            continue

        if fps_for_spacing is None:
            fps_for_spacing = fps

        step = (fps_for_spacing or 30) * 600  # ~10 min; fallback 30fps
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
    Parse the 'videos' cell from mapping.csv which can look like:
      "['vidA','vidB']" or "vidA, vidB"
    Be tolerant of formats. Prefer literal_eval when possible.
    """
    if not videos_str:
        return []

    # Try literal_eval first
    try:
        data = ast.literal_eval(videos_str)
        if isinstance(data, (list, tuple)):
            return [str(x).strip() for x in data if str(x).strip()]
        # If it's a single string, fall through to split
    except Exception:
        pass

    # Fallback: simple CSV-like split
    parts = [p.strip().strip("'").strip('"') for p in videos_str.strip("[]").split(",")]
    return [p for p in parts if p]


def get_video_mapping(mapping_csv_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Build a dictionary mapping video_id -> (city, country)
    """
    video_mapping: Dict[str, Tuple[str, str]] = {}
    try:
        with open(mapping_csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                city = row.get("city", "")
                country = row.get("country", "")
                videos_str = row.get("videos", "")
                videos_list = parse_videos_list_field(videos_str)
                for vid in videos_list:
                    if vid:
                        video_mapping[vid] = (city, country)
        logger.info("Loaded video mapping for {} entries", len(video_mapping))
    except Exception as e:
        logger.error("Could not load video mapping from {}. Error: {}", mapping_csv_path, e)
    return video_mapping


def _ffprobe_codec(video_path: str) -> Optional[str]:
    """Return codec_name of the first video stream, or None on failure."""
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
    Re-encode input to H.264/AAC with *software* decode to avoid HW AV1 errors.
    Returns path to the new file, or None on failure.
    """
    base, _ = os.path.splitext(video_path)
    out_path = base + "_reencoded.mp4"
    if os.path.exists(out_path):
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "none",        # force software decode (avoid AV1 HW errors)
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
    Try to open the video robustly:
      - If the source is AV1 (via ffprobe), transcode to H.264 with software decode first.
      - Otherwise try to open directly; if that fails, do the same fallback.
    """
    codec = _ffprobe_codec(video_path)
    if codec and codec.lower() in {"av1"}:
        reenc = _reencode_to_h264_sw(video_path)
        if reenc:
            cap = cv2.VideoCapture(reenc)
            if cap.isOpened():
                logger.info("Opened AV1 source via software reencode: {}", reenc)
                return cap

    # Try direct open
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap

    # Last-ditch: re-encode even if codec wasn't detected as AV1 (covers odd codecs/builds)
    reenc = _reencode_to_h264_sw(video_path)
    if reenc:
        cap2 = cv2.VideoCapture(reenc)
        if cap2.isOpened():
            logger.info("Opened source via fallback reencode: {}", reenc)
            return cap2

    logger.error("Could not open video (original or reencoded): {}", video_path)
    return None


def save_frames_with_mapping(
    video_path: str, frame_numbers: List[int], save_dir: str, video_mapping: Dict[str, Tuple[str, str]]
) -> None:
    """
    Save frames with filenames: {city}_{country}_{videoid}_{frame_number}.jpg
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

    city, country = video_mapping[video_id]

    for frame_num in frame_numbers:
        if frame_num < 0 or frame_num >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame {} from {}", frame_num, video_path)
            continue

        filename = f"{city}_{country}_{video_id}_{frame_num}.jpg"
        out_path = os.path.join(save_dir, filename)
        cv2.imwrite(out_path, frame)
        logger.info("Saved frame {} -> {}", frame_num, out_path)

    cap.release()


def main() -> None:
    """
    Mapping-first workflow:
      - Load configs and mapping.
      - Iterate mapping.csv in file order (row-by-row).
      - For the first (or all) rows that have a non-empty 'country':
          For each video_id in that row:
            - Find CSVs for that video_id.
            - Select frames only from those CSVs.
            - Find the .mp4 in any configured video_dirs.
            - Save frames named {city}_{country}_{videoid}_{frame}.jpg
    """
    try:
        bbox_dir = get_configs("BBOX_DIR")
        video_dirs: List[str] = get_configs("video_dirs")
        save_dir = get_configs("SAVE_DIR")

        min_persons = get_configs("MIN_PERSONS")
        min_cars = get_configs("MIN_CARS")
        min_lights = get_configs("MIN_LIGHTS")
        max_frames = get_configs("MAX_FRAMES")
        window = get_configs("CONF_WINDOW")  # e.g., 10–15

        mapping_csv_path = os.path.join(root_dir, "mapping.csv")
    except KeyError as e:
        logger.error("Missing required configuration key: {}", e)
        return
    except Exception as e:
        logger.error("Configuration loading failed. Error: {}", e)
        return

    # Build lookup: video_id -> (city, country)
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
                    # 1) CSVs for this video
                    csv_paths = glob_csvs_for_video(bbox_dir, video_id)
                    if not csv_paths:
                        logger.warning("No CSVs for video_id='{}'; skipping.", video_id)
                        continue

                    # 2) Select frames (for this video only)
                    frame_numbers = select_frames_for_csvs(
                        csv_paths, min_persons, min_cars, min_lights, max_frames, window
                    )
                    if not frame_numbers:
                        logger.info("No frames matched thresholds for video_id='{}'.", video_id)
                        continue

                    # 3) Find the actual .mp4 in any of the configured directories
                    found_video: Optional[str] = None
                    for folder in video_dirs:
                        candidate = os.path.join(folder, f"{video_id}.mp4")
                        if os.path.exists(candidate):
                            found_video = candidate
                            break

                    if not found_video:
                        logger.error("Video file for ID '{}' not found in any directory.", video_id)
                        continue

                    # 4) Save frames with mapping metadata in filename
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
