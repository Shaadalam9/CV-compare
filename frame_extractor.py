# by Md Shadab Alam <md_shadab_alam@outlook.com>
"""
Frame extraction from videos based on YOLO-detected objects (mapping-first, AV1-robust).

Overview
--------
This script selects and extracts representative frames from a set of videos, using
pre-computed YOLO detection CSVs as guidance. It emphasizes *mapping-first* operation:
each video is tied to a {city, country, time-of-day} mapping row from a `mapping.csv`
so that saved frames can be named and organized consistently.

Why this exists
---------------
- Efficiently sample frames that contain a minimum presence of people, cars, and traffic lights.
- Ensure *temporal stability* via a rolling window so frames are not isolated detections.
- Handle AV1 and other OpenCV-incompatible codecs by re-encoding automatically.
- Save outputs with metadata-friendly filenames and day/night subfolders.

Expected inputs
---------------
1) **YOLO CSVs** (one per segment):
   Filenames follow: `{video_id}_{start_time}_{fps}.csv`
   Example: `abc123_45_30.csv` => video_id="abc123", start_time=45 seconds, fps=30 FPS.

   Minimal columns required (names are case-sensitive):
   - `frame-count`  (int): frame index *within the segment*
   - `yolo-id`      (int): class index per detection (0=person, 2=car, 9=traffic light)

   Other columns can exist (e.g. `unique-id`, `confidence`), and may be messy.
   This script only *needs* the two columns above and handles type issues robustly.

2) **mapping.csv** (row per city/country with one or many videos):
   Required columns:
   - `city` (str)
   - `country` (str, non-empty rows are processed)
   - `videos` (list-like string or comma-separated): e.g. "['abc','def']" or "abc,def"
   Optional (for day/night segmentation):
   - `time_of_day` : nested list of 0/1 per video (0=day, 1=night); supports multi segments
   - `start_time`  : nested list of frame-start (absolute frame index in full video)
   - `end_time`    : nested list of frame-end   (absolute frame index in full video)

Outputs
-------
JPEG frames saved as:
    {SAVE_DIR}/{day|night}/{city}_{country}_{videoid}_{frame_number}.jpg

Robustness highlights
---------------------
- Polars CSV read is *pinned* for the troublesome `unique-id` column (frequently floats)
  to avoid "could not parse as i64" errors. We only rely on `frame-count` & `yolo-id`.
- Deeper inference window + permissive fallback with `ignore_errors=True`.
- AV1-aware video open path: probe codec with ffprobe; re-encode via ffmpeg to H.264
  when needed; fall back gracefully.

Configuration (via `common.get_configs(key)`)
---------------------------------------------
- `BBOX_DIR` (str): folder with the YOLO CSVs
- `video_dirs` (list[str]): directories to search for `{video_id}.mp4`
- `SAVE_DIR` (str): root output folder for extracted frames
- `MIN_PERSONS`, `MIN_CARS`, `MIN_LIGHTS` (int): detection thresholds per frame
- `MAX_FRAMES` (int): hard cap for selected frames per video
- `CONF_WINDOW` (int): consecutive frames window where thresholds must hold (stability)
- `frame_interval` (int|float): spacing (in seconds) between *selected* frames
- `mapping` (str): path to mapping.csv
"""

from __future__ import annotations

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

# -----------------------------------------------------------------------------
# Logging & toggles
# -----------------------------------------------------------------------------

logger = CustomLogger(__name__)

# If True, stop after processing the first mapping.csv row that has a non-empty 'country'.
STOP_AFTER_FIRST_VALID_MAPPING_ROW = False

# -----------------------------------------------------------------------------
# Constants (YOLO class IDs used)
# -----------------------------------------------------------------------------
YOLO_PERSON = 0
YOLO_CAR = 2
YOLO_TRAFFIC_LIGHT = 9


# -----------------------------------------------------------------------------
# CSV reading utilities
# -----------------------------------------------------------------------------
def _read_yolo_csv(csv_path: str) -> Optional[pl.DataFrame]:
    """
    Read a YOLO detection CSV robustly with Polars.

    Why a custom reader?
    --------------------
    Polars infers schema from an initial sample. In many datasets, a column like
    `unique-id` looks like an integer early on but contains decimals later (e.g. 0.817904).
    If that column is inferred as Int64 (`i64`), parsing fails when it meets a float.

    Strategy
    --------
    1) Pin `unique-id` => Float64 (or Utf8 in fallback), so mixed numeric forms parse.
    2) Explicitly set the types we *need* (`frame-count`, `yolo-id`) to Int64.
    3) Use deeper `infer_schema_length` to reduce mis-inference for other columns.
    4) On failure, fall back to permissive read with `ignore_errors=True` and coerce
       our two required columns.

    Parameters
    ----------
    csv_path : str
        Full path to the CSV file.

    Returns
    -------
    Optional[pl.DataFrame]
        A DataFrame on success; None if both primary and fallback reads fail.
    """
    # Explicit overrides for known-problem and required columns.
    schema_overrides = {
        "unique-id": pl.Float64,  # avoid int parsing errors when decimals appear
        "frame-count": pl.Int64,  # required column
        "yolo-id": pl.Int64,      # required column
    }

    try:
        df = pl.read_csv(
            csv_path,
            infer_schema_length=10000,  # peek deeper before deciding dtypes
            schema_overrides=schema_overrides,
        )
        # Ensure required columns are ints even if they arrived as strings.
        df = df.with_columns(
            pl.col("frame-count").cast(pl.Int64, strict=False),
            pl.col("yolo-id").cast(pl.Int64, strict=False),
        )
        return df
    except Exception as e:
        logger.error("Unable to read CSV file {}. Error: {}", csv_path, e)

    # Fallback: be permissive; we only need 2 columns to proceed.
    try:
        df = pl.read_csv(
            csv_path,
            infer_schema_length=10000,
            schema_overrides={"unique-id": pl.Utf8},  # most permissive; keep as text
            ignore_errors=True,  # only offending *fields* in rows are skipped
        ).with_columns(
            pl.col("frame-count").cast(pl.Int64, strict=False),
            pl.col("yolo-id").cast(pl.Int64, strict=False),
        )
        return df
    except Exception as e:
        logger.error("Fallback read also failed for {}. Error: {}", csv_path, e)
        return None


# -----------------------------------------------------------------------------
# Frame selection per CSV (segment) & across CSVs (full video)
# -----------------------------------------------------------------------------
def find_frames_with_real_index(
    csv_path: str,
    min_persons: int,
    min_cars: int,
    min_lights: int,
    window: int = 10
) -> Tuple[str, int, pl.DataFrame]:
    """
    Identify stable, threshold-meeting frames from a YOLO CSV segment and convert to real indices.

    The CSV filename format is `{video_id}_{start}_{fps}.csv`. We:
      1) Parse that filename to recover `video_id`, `segment_start_secs`, and `fps`.
      2) Group detections by `frame-count` and count target classes.
      3) Mark frames that meet thresholds (persons >= min_persons, etc.).
      4) Enforce *temporal stability*: a frame is valid only if the condition holds for
         `window` consecutive frames (rolling minimum on the boolean condition).
      5) Convert segment-relative `frame-count` to absolute `real-frame` by offsetting with
         `segment_start_secs * fps`.

    Parameters
    ----------
    csv_path : str
        Path to YOLO detection CSV (single segment of a video).
    min_persons : int
        Minimum people to consider a frame.
    min_cars : int
        Minimum cars to consider a frame.
    min_lights : int
        Minimum traffic lights to consider a frame.
    window : int, default=10
        Number of consecutive frames that must satisfy the thresholds to be considered stable.

    Returns
    -------
    (video_id, fps, valid_frames_df) : Tuple[str, int, pl.DataFrame]
        - video_id (str) : extracted from the filename.
        - fps (int)      : extracted from the filename.
        - valid_frames_df: columns: ["frame-count","persons","cars","traffic_lights",
                                     "criteria_met","stable_window","real-frame"]
          Only rows with stable_window==True are returned.
    """
    # Parse filename tokens.
    filename = os.path.basename(csv_path)
    match = re.match(r"(.+?)_(\d+)_(\d+)\.csv", filename)
    if not match:
        logger.warning("Skipped CSV due to unexpected filename format: {}", filename)
        return "", 0, pl.DataFrame()

    video_id, start_time_str, fps_str = match.groups()
    start_time, fps = int(start_time_str), int(fps_str)

    # Read CSV with robust parser
    df = _read_yolo_csv(csv_path)
    if df is None or df.is_empty():
        return video_id, fps, pl.DataFrame()

    # Sanity check: required columns exist
    required_cols = {"frame-count", "yolo-id"}
    if not required_cols.issubset(set(df.columns)):
        logger.error("CSV {} missing required columns {}; skipping.", csv_path, required_cols - set(df.columns))
        return video_id, fps, pl.DataFrame()

    # Aggregate counts per frame-count for classes of interest
    grouped = (
        df.group_by("frame-count")
        .agg([
            (pl.col("yolo-id") == YOLO_PERSON).sum().alias("persons"),
            (pl.col("yolo-id") == YOLO_CAR).sum().alias("cars"),
            (pl.col("yolo-id") == YOLO_TRAFFIC_LIGHT).sum().alias("traffic_lights"),
        ])
        .sort("frame-count")
        # Per-frame threshold check
        .with_columns(
            ((pl.col("persons") >= min_persons)
             & (pl.col("cars") >= min_cars)
             & (pl.col("traffic_lights") >= min_lights)
             ).alias("criteria_met")
        )
        # Temporal stability: boolean rolling minimum across `window` frames
        .with_columns(
            pl.col("criteria_met").rolling_min(window_size=window, min_samples=window).alias("stable_window")
        )
    )

    # Only keep frames within stable spans
    valid_frames = grouped.filter(pl.col("stable_window").eq(True))

    # Convert segment-local frame number to real (absolute within full video)
    offset = start_time * fps
    valid_frames = valid_frames.with_columns((pl.col("frame-count") + offset).alias("real-frame"))

    return video_id, fps, valid_frames


def glob_csvs_for_video(bbox_dir: str, video_id: str) -> List[str]:
    """
    Find all YOLO CSV segments for a given video in a directory.

    Parameters
    ----------
    bbox_dir : str
        Directory containing YOLO CSV files.
    video_id : str
        Target video ID.

    Returns
    -------
    List[str]
        Sorted list of CSV filepaths matching the pattern `{video_id}_*_*.csv`.
        Empty if none found.
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
    Aggregate and select valid, *evenly spaced* absolute frame numbers across all segments of a video.

    How spacing works
    -----------------
    We compute a spacing step (in frames) using:
        step = fps_for_spacing * common.get_configs("frame_interval")
    where `fps_for_spacing` is taken from the first segment with detections,
    and `frame_interval` is a config (in *seconds*). This ensures roughly one
    frame sampled per given time interval, but only from periods that satisfy
    the thresholds + stability criteria.

    Parameters
    ----------
    csv_paths : List[str]
        Segment CSV filepaths for the same video.
    min_persons, min_cars, min_lights : int
        Detection thresholds per frame.
    max_frames : int
        Hard cap on the number of frames to return across all segments.
    window : int
        Temporal stability window (consecutive frames).

    Returns
    -------
    List[int]
        Sorted list of *absolute* frame indices (`real-frame`) to extract.
        Empty if none qualified.
    """
    if not csv_paths:
        return []

    collected: List[int] = []
    fps_for_spacing: Optional[int] = None  # determined once from the first valid segment

    for csv_path in csv_paths:
        _, fps, valid_frames_df = find_frames_with_real_index(csv_path, min_persons, min_cars, min_lights, window)
        if valid_frames_df.is_empty():
            logger.info("No valid frames in CSV: {}", csv_path)
            continue

        if fps_for_spacing is None:
            fps_for_spacing = fps

        # Convert desired time spacing (seconds) to frames
        step = (fps_for_spacing or 30) * common.get_configs("frame_interval")  # pyright: ignore[reportOperatorIssue]
        next_target = collected[-1] + step if collected else 0

        # Greedy selection: pick the earliest valid frame >= next_target, then advance
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


# -----------------------------------------------------------------------------
# Mapping.csv parsing (single and multi-segment day/night support)
# -----------------------------------------------------------------------------
def parse_videos_list_field(videos_str: str) -> List[str]:
    """
    Parse and normalize the 'videos' field from mapping.csv into a clean list of IDs.

    Accepts two common formats:
      - Python-list-like string: "['vidA','vidB']"
      - Comma-separated: "vidA, vidB"

    Parameters
    ----------
    videos_str : str

    Returns
    -------
    List[str]
        List of video IDs, stripped of quotes and whitespace.
    """
    if not videos_str:
        return []

    # Try literal_eval first for true list syntax
    try:
        data = ast.literal_eval(videos_str)
        if isinstance(data, (list, tuple)):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    # Fallback: split comma-separated tokens, clean quotes
    parts = [p.strip().strip("'").strip('"') for p in videos_str.strip("[]").split(",")]
    return [p for p in parts if p]


def build_time_interval_mapping(row: dict) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Build a mapping of video_id -> list of (start_frame, end_frame, time_of_day) for multi-segment videos.

    This is used when a row indicates the presence of both day(0) and night(1) intervals for a video.
    The three fields `time_of_day`, `start_time`, `end_time` are expected to be *nested lists*, each
    aligned per video (e.g., outer dimension is per-video, inner lists are per-interval).

    Example (conceptual)
    --------------------
    row["videos"]      : "['vidA']"
    row["time_of_day"] : "[[0, 1]]"
    row["start_time"]  : "[[  0, 3000]]"
    row["end_time"]    : "[[2999, 6000]]"

    Returns
    -------
    Dict[str, List[Tuple[int, int, int]]]
        { video_id: [(start_frame, end_frame, tod), ...], ... }
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

                # Normalize to lists if scalar
                if not isinstance(tod_list, list):
                    tod_list = [tod_list]
                if not isinstance(start_list, list):
                    start_list = [start_list]
                if not isinstance(end_list, list):
                    end_list = [end_list]

                intervals: List[Tuple[int, int, int]] = []
                for j, tod in enumerate(tod_list):
                    try:
                        s = int(start_list[j])
                        e = int(end_list[j])
                        intervals.append((s, e, int(tod)))
                    except Exception:
                        # Skip misaligned sub-entries gracefully
                        continue

                if intervals:
                    mapping[vid] = intervals
            except Exception:
                # Skip malformed per-video entries gracefully
                continue
    except Exception as e:
        logger.warning("Failed to build interval mapping: {}", e)
    return mapping


def get_video_mapping(mapping_csv_path) -> Dict[str, Tuple[str, str, object]]:
    """
    Load and parse mapping.csv into: {video_id: (city, country, tod_info)}

    `tod_info` can be:
      - int: 0 (day) or 1 (night), if the video is single-period.
      - list[tuple]: [(start, end, tod), ...] for multi-segment day/night splits.

    Logic
    -----
    - Detect if a row contains both 0 and 1 in its nested `time_of_day`.
      If yes, build explicit intervals via `build_time_interval_mapping`.
    - Otherwise, assign a single tod (0 or 1) per video (defaults to day=0).

    Parameters
    ----------
    mapping_csv_path : str

    Returns
    -------
    Dict[str, Tuple[str, str, object]]
        video_id -> (city, country, tod_info)
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
                nested: List[List[int]] = []

                # Parse nested time_of_day like [[0], [1]] or [[0,1]] etc.
                try:
                    nested = ast.literal_eval(time_field)
                    flattened = [item for sublist in nested for item in sublist if isinstance(item, int)]
                    if 0 in flattened and 1 in flattened:
                        time_of_day = [0, 1]  # indicates mixed periods exist in this row
                    elif 1 in flattened:
                        time_of_day = 1
                    else:
                        time_of_day = 0
                except Exception:
                    time_of_day = 0
                    nested = []

                # If a row declares both day and night, we treat it as multi-segment
                if isinstance(time_of_day, list) and 0 in time_of_day and 1 in time_of_day:
                    interval_mapping = build_time_interval_mapping(row)
                    for vid, intervals in interval_mapping.items():
                        video_mapping[vid] = (city, country, intervals)
                    continue

                # Otherwise, assign single tod per video based on whether any '1' is present
                for i, vid in enumerate(videos_list):
                    tod = 0
                    try:
                        if i < len(nested):
                            # If any '1' exists in the ith sublist, treat entire video as night=1
                            if 1 in [int(x) for x in nested[i] if isinstance(x, int)]:
                                tod = 1
                    except Exception:
                        tod = 0
                    video_mapping[vid] = (city, country, tod)

        logger.info("Loaded video mapping for {} entries", len(video_mapping))
    except Exception as e:
        logger.error("Could not load video mapping from {}. Error: {}", mapping_csv_path, e)

    return video_mapping


# -----------------------------------------------------------------------------
# Video IO (codec probing and safe open)
# -----------------------------------------------------------------------------
def _ffprobe_codec(video_path: str) -> Optional[str]:
    """
    Retrieve the codec name (e.g. 'h264', 'av1') for the first video stream via ffprobe.

    Returns
    -------
    Optional[str]
        Codec name lowercase, or None if probing fails.
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
    Re-encode video to H.264 using *software* decoding to ensure OpenCV compatibility.

    Why?
    ----
    Many OpenCV builds cannot decode AV1, and some hardware-encoded streams are flaky.
    We force software decode and output a yuv420p H.264 file for maximum compatibility.

    Returns
    -------
    Optional[str]
        Re-encoded path if successful; None otherwise.
    """
    base, _ = os.path.splitext(video_path)
    out_path = base + "_reencoded.mp4"
    if os.path.exists(out_path):
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "none",   # ensure software path
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
    Open a video robustly, re-encoding to H.264 if needed.

    Steps
    -----
    1) ffprobe the codec; if AV1 -> re-encode first.
    2) Try to open original; if fails -> re-encode and open the copy.
    3) Return an opened cv2.VideoCapture or None on failure.
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


# -----------------------------------------------------------------------------
# Frame extraction & saving
# -----------------------------------------------------------------------------
def save_frames_with_mapping(
    video_path: str,
    frame_numbers: List[int],
    save_dir: str,
    video_mapping: Dict[str, Tuple[str, str, object]]
) -> None:
    """
    Extract and save given absolute frame numbers to day/night subfolders with city/country names.

    Behavior
    --------
    - Determines the `video_id` from the filename (without extension).
    - Looks up (city, country, tod_info) from the pre-built mapping.
    - If `tod_info` is an int: save all frames under {save_dir}/{day|night}/
    - If `tod_info` is a list of (start,end,tod): route frames into appropriate segments.

    Parameters
    ----------
    video_path : str
        Full path to the source video file (`{video_id}.mp4`).
    frame_numbers : List[int]
        Absolute frame indices to extract.
    save_dir : str
        Root output folder.
    video_mapping : Dict[str, Tuple[str, str, object]]
        video_id -> (city, country, tod_info)
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
        # Single TOD per video (0=day, 1=night)
        time_label = "night" if tod_info == 1 else "day"
        subdir = os.path.join(save_dir, time_label)
        os.makedirs(subdir, exist_ok=True)

        saved = 0
        for frame_no in frame_numbers:
            if frame_no >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, frame = cap.read()
            if success:
                out_name = f"{city}_{country}_{video_id}_{frame_no}.jpg"
                cv2.imwrite(os.path.join(subdir, out_name), frame)
                saved += 1
        cap.release()
        logger.info("Saved {} frames for {} ({})", saved, video_id, time_label)
        return

    # Multi-segment day/night (route frames by interval)
    for start, end, tod in tod_info:
        label = "night" if tod == 1 else "day"
        subdir = os.path.join(save_dir, label)
        os.makedirs(subdir, exist_ok=True)

        # Keep only frames within this segment’s interval
        segment_frames = [f for f in frame_numbers if start <= f <= end]

        saved = 0
        for frame_no in segment_frames:
            if frame_no >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, frame = cap.read()
            if success:
                out_name = f"{city}_{country}_{video_id}_{frame_no}.jpg"
                cv2.imwrite(os.path.join(subdir, out_name), frame)
                saved += 1

        logger.info("Saved {} frames for {} ({}) segment", saved, video_id, label)

    cap.release()


# -----------------------------------------------------------------------------
# Orchestration (main)
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Orchestrate the complete mapping-first frame extraction workflow.

    Steps
    -----
    1) Read configs (paths, thresholds, limits).
    2) Build `video_mapping` from mapping.csv (city, country, tod info per video).
    3) Iterate mapping.csv rows with non-empty `country`:
       a) For each video listed in the row:
          - Find segment CSVs (YOLO detections).
          - Compute valid absolute frame numbers (threshold + stability).
          - Locate the real video file in one of the configured `video_dirs`.
          - Extract and save frames under {SAVE_DIR}/{day|night}/ with descriptive names.
    4) Optionally stop after first valid row if `STOP_AFTER_FIRST_VALID_MAPPING_ROW` is True.

    Notes
    -----
    - All errors are logged and the pipeline continues on a best-effort basis.
    - Ensure the `{video_id}` in CSV filenames matches `{video_id}.mp4` names.
    """
    # -- Read configuration keys safely
    try:
        bbox_dir = common.get_configs("BBOX_DIR")
        video_dirs = common.get_configs("video_dirs")
        save_dir: str = common.get_configs("SAVE_DIR")  # type: ignore
        min_persons: int = common.get_configs("MIN_PERSONS")  # type: ignore
        min_cars: int = common.get_configs("MIN_CARS")  # type: ignore
        min_lights: int = common.get_configs("MIN_LIGHTS")  # type: ignore
        max_frames: int = common.get_configs("MAX_FRAMES")  # type: ignore
        window: int = common.get_configs("CONF_WINDOW")  # type: ignore
        mapping_csv_path = common.get_configs("mapping")
    except KeyError as e:
        logger.error("Missing required configuration key: {}", e)
        return
    except Exception as e:
        logger.error("Configuration loading failed. Error: {}", e)
        return

    # -- Build video mapping (city, country, TOD segmentation)
    video_mapping = get_video_mapping(mapping_csv_path)
    if not video_mapping:
        logger.error("Empty or unreadable mapping.csv; nothing to do.")
        return

    processed_any_row = False

    # -- Iterate the mapping.csv for actual work
    try:
        with open(mapping_csv_path, newline="", encoding="utf-8") as csvfile:  # type: ignore
            reader = csv.DictReader(csvfile)

            for row in reader:
                city = row.get("city", "")
                country = row.get("country", "")
                videos_str = row.get("videos", "")

                # Skip rows that do not identify a country (as per spec)
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

                # Process each video in this row
                for video_id in videos_list:
                    # Find all CSV segments for this video
                    csv_paths = glob_csvs_for_video(bbox_dir, video_id)  # type: ignore
                    if not csv_paths:
                        logger.warning("No CSVs for video_id='{}'; skipping.", video_id)
                        continue

                    # Compute frame numbers that meet thresholds + stability, spaced by frame_interval
                    frame_numbers = select_frames_for_csvs(
                        csv_paths, min_persons, min_cars, min_lights, max_frames, window
                    )
                    if not frame_numbers:
                        logger.info("No frames matched thresholds for video_id='{}'.", video_id)
                        continue

                    # Find the actual .mp4 under any configured video directory
                    found_video: Optional[str] = None
                    for folder in video_dirs:  # type: ignore
                        candidate = os.path.join(folder, f"{video_id}.mp4")
                        if os.path.exists(candidate):
                            found_video = candidate
                            break

                    if not found_video:
                        logger.error("Video file for ID '{}' not found in any directory.", video_id)
                        continue

                    # Extract frames and save with mapping
                    save_frames_with_mapping(found_video, frame_numbers, save_dir, video_mapping)

                processed_any_row = True

                if STOP_AFTER_FIRST_VALID_MAPPING_ROW:
                    logger.info("STOP_AFTER_FIRST_VALID_MAPPING_ROW=True — stopping after this row.")
                    break

        if not processed_any_row:
            logger.warning("No mapping rows with a non-empty 'country' were processed.")
    except Exception as e:
        logger.error("Failed while iterating mapping.csv rows. Error: {}", e)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
