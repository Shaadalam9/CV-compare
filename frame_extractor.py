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
import ast

logger = CustomLogger(__name__)

# Toggle: if True, the script stops after the FIRST mapping row with a non-empty 'country'.
STOP_AFTER_FIRST_VALID_MAPPING_ROW = False


def find_frames_with_real_index(csv_path: str, min_persons: int, min_cars: int, min_lights: int, window: int = 10
                                ) -> Tuple[str, int, pl.DataFrame]:
    """
    Identify valid frame indices from a YOLO detection CSV file, based on object-count thresholds
    and continuous detection stability across a time window.

    This function processes YOLO detection results saved in CSV format for a specific video clip.
    Each CSV filename encodes the video ID, start time, and FPS. The function aggregates object
    counts per frame (for person, car, and traffic light detections), applies thresholds to determine
    which frames meet criteria, and then enforces that these conditions are maintained for a 
    continuous window of frames. It converts clip-relative frame indices to absolute indices 
    relative to the full video.

    Args:
        csv_path (str): Path to the YOLO detection CSV file. 
            Expected filename format: `{video_id}_{start_time}_{fps}.csv`.
        min_persons (int): Minimum number of detected persons required per frame.
        min_cars (int): Minimum number of detected cars required per frame.
        min_lights (int): Minimum number of detected traffic lights required per frame.
        window (int, optional): Number of consecutive frames that must all meet thresholds
            to qualify as a stable detection window. Defaults to 10.

    Returns:
        Tuple[str, int, pl.DataFrame]:
            - video_id (str): Identifier of the processed video.
            - fps (int): Frames per second for the video segment.
            - valid_frames_df (pl.DataFrame): DataFrame containing frames that meet
              stability criteria, including:
                  * frame-count (relative frame number)
                  * persons, cars, traffic_lights (object counts)
                  * criteria_met (boolean for threshold satisfaction)
                  * stable_window (boolean for continuous stability)
                  * real-frame (absolute frame number in full video timeline)

    Raises:
        None. Errors are logged, and empty results are returned on failure.

    Example:
        >>> vid, fps, frames = find_frames_with_real_index("vid123_60_30.csv", 1, 2, 1, window=15)
        >>> print(frames.head())
        ┌─────────────┬────────┬──────┬────────────────┬───────────────┬────────────┐
        │ frame-count │ persons│ cars │ traffic_lights │ criteria_met  │ real-frame │
        ├─────────────┼────────┼──────┼────────────────┼───────────────┼────────────┤
        │ 250         │ 2      │ 3    │ 1              │ True          │ 2150       │
        └─────────────┴────────┴──────┴────────────────┴───────────────┴────────────┘
    """
    # Extract filename and parse video_id, start_time, and fps
    filename = os.path.basename(csv_path)
    match = re.match(r"(.+?)_(\d+)_(\d+)\.csv", filename)
    if not match:
        # Skip files that don't match the expected pattern
        logger.warning("Skipped CSV due to unexpected filename format: {}", filename)
        return "", 0, pl.DataFrame()

    video_id, start_time_str, fps_str = match.groups()
    start_time, fps = int(start_time_str), int(fps_str)

    # Attempt to read YOLO detection results
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        logger.error("Unable to read CSV file {}. Error: {}", csv_path, e)
        return video_id, fps, pl.DataFrame()

    # Group detections by frame-count and count object types of interest
    grouped = (
        df.group_by("frame-count")
        .agg([
            (pl.col("yolo-id") == 0).sum().alias("persons"),          # COCO ID 0 = person
            (pl.col("yolo-id") == 2).sum().alias("cars"),             # COCO ID 2 = car
            (pl.col("yolo-id") == 9).sum().alias("traffic_lights"),   # COCO ID 9 = traffic light
        ])
        .sort("frame-count")
        .with_columns(
            # Determine if each frame meets all threshold criteria
            (
                (pl.col("persons") >= min_persons)
                & (pl.col("cars") >= min_cars)
                & (pl.col("traffic_lights") >= min_lights)
            ).alias("criteria_met")
        )
        .with_columns(
            # Enforce that criteria are met for a continuous rolling window of frames
            pl.col("criteria_met")
            .rolling_min(window_size=window, min_samples=window)
            .alias("stable_window")
        )
    )

    # Keep only frames that satisfy the stable window condition
    valid_frames = grouped.filter(pl.col("stable_window").eq(True))

    # Convert clip-relative frame-counts to absolute (real) video frame indices
    offset = start_time * fps
    valid_frames = valid_frames.with_columns(
        (pl.col("frame-count") + offset).alias("real-frame")
    )

    return video_id, fps, valid_frames


def glob_csvs_for_video(bbox_dir: str, video_id: str) -> List[str]:
    """
    Retrieve all YOLO detection CSV files associated with a specific video ID.

    This function searches within a given bounding box directory (`bbox_dir`) for all
    CSV files that belong to a particular `video_id`. The expected filename pattern is:
    `{video_id}_{start_time}_{fps}.csv`, where `start_time` represents the segment start
    time (in seconds) and `fps` is the frames per second of that segment.

    Args:
        bbox_dir (str): Directory path containing YOLO-generated bounding box CSV files.
        video_id (str): Unique identifier of the target video.

    Returns:
        List[str]: A sorted list of matching CSV file paths for the specified `video_id`.
        Returns an empty list if no matches are found.

    Raises:
        None. A warning is logged if no CSV files are found.

    Example:
        >>> files = glob_csvs_for_video("/data/bbox_csvs", "vid123")
        >>> print(files)
        ['/data/bbox_csvs/vid123_0_30.csv', '/data/bbox_csvs/vid123_60_30.csv']
    """
    # Construct the expected filename pattern, e.g., "vid123_60_30.csv"
    pattern = os.path.join(bbox_dir, f"{video_id}_*_*.csv")

    # Retrieve all matching CSV file paths and sort them by filename
    paths = sorted(glob.glob(pattern))

    # Log a warning if no files were found for this video
    if not paths:
        logger.warning("No CSVs found for video_id='{}' in {}", video_id, bbox_dir)

    return paths


def select_frames_for_csvs(csv_paths: List[str], min_persons: int, min_cars: int, min_lights: int, max_frames: int,
                           window: int) -> List[int]:
    """
    Aggregate valid absolute frame indices for a single video across multiple YOLO CSV files.

    This function iterates over all CSV files belonging to a single video (each representing
    a time segment of that video), extracts frame indices that meet detection thresholds
    using `find_frames_with_real_index()`, and collects a limited number of frames spaced
    approximately 10 minutes apart to ensure temporal diversity.

    The goal is to identify representative frames across the entire video where detection
    criteria (persons, cars, traffic lights) are consistently met over a continuous window.

    Args:
        csv_paths (List[str]): List of YOLO detection CSV file paths for a given video.
            Each CSV should follow the naming pattern `{video_id}_{start_time}_{fps}.csv`.
        min_persons (int): Minimum number of detected persons required per frame.
        min_cars (int): Minimum number of detected cars required per frame.
        min_lights (int): Minimum number of detected traffic lights required per frame.
        max_frames (int): Maximum number of frames to extract across all CSVs.
        window (int): Number of consecutive frames that must meet the detection thresholds.

    Returns:
        List[int]: Sorted list of absolute frame indices (`real-frame` values)
        selected from the combined CSV files. The frames are spaced roughly
        10 minutes apart in the video timeline.

    Raises:
        None. Errors are logged internally.

    Example:
        >>> csvs = [
        ...     "/data/bbox_csvs/vid001_0_30.csv",
        ...     "/data/bbox_csvs/vid001_600_30.csv",
        ... ]
        >>> frames = select_frames_for_csvs(csvs, 1, 2, 1, max_frames=5, window=10)
        >>> print(frames)
        [1250, 18650, 36450, 54650, 72250]
    """
    # Return immediately if no CSV files are provided
    if not csv_paths:
        return []

    collected: List[int] = []       # List to store selected real-frame indices
    fps_for_spacing: Optional[int] = None  # Used to calculate 10-minute frame spacing

    # Iterate through each YOLO CSV belonging to this video
    for csv_path in csv_paths:
        # Extract valid frames for this CSV based on detection thresholds and window stability
        _, fps, valid_frames_df = find_frames_with_real_index(
            csv_path, min_persons, min_cars, min_lights, window
        )

        # Skip CSVs that produce no valid frames
        if valid_frames_df.is_empty():
            logger.info("No valid frames in CSV: {}", csv_path)
            continue

        # Capture FPS from the first usable CSV to determine spacing between frames
        if fps_for_spacing is None:
            fps_for_spacing = fps

        # Compute step interval (~10 minutes apart); default to 30 FPS if unknown
        # 600 seconds * fps = 10-minute interval
        step = (fps_for_spacing or 30) * common.get_configs("frame_interval")  # type: ignore
        next_target = collected[-1] + step if collected else 0  # First frame starts at 0

        # Iterate through valid frames (already absolute frame numbers)
        for row in valid_frames_df.iter_rows(named=True):
            rf = row["real-frame"]

            # Only add frame if it meets the next temporal target (i.e., spaced apart)
            if not collected or rf >= next_target:
                collected.append(rf)
                next_target = rf + step  # Update target for next 10-minute interval

            # Stop early if we’ve reached the frame extraction limit
            if len(collected) >= max_frames:
                break

        # Break out once max_frames limit is reached across all CSVs
        if len(collected) >= max_frames:
            break

    logger.info("Selected {} frames for this video", len(collected))
    return collected


def parse_videos_list_field(videos_str: str) -> List[str]:
    """
    Parse and normalize the 'videos' field from a mapping CSV row into a list of video IDs.

    The 'videos' field in `mapping.csv` may contain video identifiers in different formats:
    - A Python-like list string (e.g., `"['vidA', 'vidB']"`)
    - A comma-separated string (e.g., `"vidA, vidB"`)
    - A single video ID string (e.g., `"vidA"`)

    This function attempts to interpret the field flexibly by first using Python's
    `ast.literal_eval()` for structured input, and then falling back to a manual
    split for less structured data. It trims extraneous characters such as quotes,
    brackets, and whitespace.

    Args:
        videos_str (str): The raw string value of the 'videos' cell from `mapping.csv`.

    Returns:
        List[str]: A cleaned list of video IDs extracted from the input string.
        Returns an empty list if the input is empty or invalid.

    Raises:
        None. All parsing errors are handled gracefully.

    Example:
        >>> parse_videos_list_field("['vidA','vidB']")
        ['vidA', 'vidB']
        >>> parse_videos_list_field("vidA, vidB")
        ['vidA', 'vidB']
        >>> parse_videos_list_field(" vidC ")
        ['vidC']
    """
    # Return empty list if no input string is provided
    if not videos_str:
        return []

    # Attempt to safely evaluate structured Python-like lists (e.g., "['vidA', 'vidB']")
    try:
        data = ast.literal_eval(videos_str)
        if isinstance(data, (list, tuple)):
            # Clean and filter out any empty or whitespace-only entries
            return [str(x).strip() for x in data if str(x).strip()]
        # If it's a single string (e.g., "'vidA'"), fall through to the manual parsing step
    except Exception:
        # Ignore parsing errors and proceed with fallback
        pass

    # Fallback: handle simple comma-separated or bracket-wrapped strings manually
    # Example: "vidA, vidB" or "[vidA, vidB]"
    parts = [
        p.strip().strip("'").strip('"')
        for p in videos_str.strip("[]").split(",")
    ]

    # Return only non-empty cleaned entries
    return [p for p in parts if p]


def get_video_mapping(mapping_csv_path) -> Dict[str, Tuple[str, str,int]]:
    """
    Load and construct a dictionary mapping each video ID to its associated city and country.

    This function reads the `mapping.csv` file, which defines the relationship between
    geographic locations and video identifiers. Each row typically contains:
      - `city`: The city name associated with the videos.
      - `country`: The corresponding country name.
      - `videos`: A string containing one or more video IDs (e.g., "['vidA','vidB']" or "vidA, vidB").

    It uses `parse_videos_list_field()` to handle flexible formats of the `videos` field,
    ensuring that all listed video IDs are extracted and mapped properly.

    Args:
        mapping_csv_path (str): Path to the mapping CSV file (e.g., `/data/mapping.csv`).

    Returns:
        Dict[str, Tuple[str, str]]:
            A dictionary mapping each `video_id` (str) to a tuple of (`city`, `country`).
            Example:
                {
                    "vidA": ("Paris", "France"),
                    "vidB": ("Berlin", "Germany"),
                }

            Returns an empty dictionary if the CSV cannot be read or is empty.

    Raises:
        None. Any file or parsing errors are logged, not raised.

    Example:
        >>> get_video_mapping("mapping.csv")
        {'vid001': ('London', 'UK'), 'vid002': ('Paris', 'France')}
    """
    video_mapping: Dict[str, Tuple[str, str,int]] = {}

    try:
        # Open the mapping CSV with UTF-8 encoding
        with open(mapping_csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            # Iterate through each mapping row
            for row in reader:
                city = row.get("city", "")
                country = row.get("country", "")
                videos_str = row.get("videos", "")
                # Default to 0 if not provided or invalid
                # Extract all numeric values from the time_of_day field (handles nested lists like [[0],[1,0],[0]])
                time_field = row.get("time_of_day", "")
                
                time_of_day = 0
                

                time_field = row.get("time_of_day", "")
                time_of_day = 0  # default to day

                try:
                    nested = ast.literal_eval(time_field)  # e.g., [[0],[0,1],[1]]
                      # convert string "[[0],[0,1]]" → [[0],[0,1]]
                    # Flatten just this row and check if any 1 exists
                    flattened = [item for sublist in nested for item in sublist if isinstance(item, int)]
                    if 1 in flattened:
                        time_of_day = 1
                except Exception:
                    time_of_day = 0
                
                # Parse and normalize the videos field into a list of IDs
                videos_str = row.get("videos", "")
                videos_list = parse_videos_list_field(videos_str)

                # Map each video ID to its (city, country) pair
                videos_list = parse_videos_list_field(videos_str)
                for i, vid in enumerate(videos_list):
                    tod = 0
                    try:
                        if i < len(nested):
                            # If any 1 in that sublist, mark night
                            if 1 in [int(x) for x in nested[i] if isinstance(x, int)]:
                                tod = 1
                    except Exception:
                        tod = 0
                    video_mapping[vid] = (city, country, tod)

        # Log the total number of mappings created
        logger.info("Loaded video mapping for {} entries", len(video_mapping))

    except Exception as e:
        # Log file or parsing errors
        logger.error(
            "Could not load video mapping from {}. Error: {}", mapping_csv_path, e
        )

    return video_mapping


def _ffprobe_codec(video_path: str) -> Optional[str]:
    """
    Retrieve the codec name of the first video stream in a media file using FFprobe.

    This helper function executes the `ffprobe` command-line tool to inspect the given
    video file and extract the codec name (e.g., "h264", "av1", "hevc") of its primary
    video stream. It is used to detect whether a video requires re-encoding (for example,
    to handle AV1 codec compatibility issues).

    Args:
        video_path (str): Path to the input video file.

    Returns:
        Optional[str]: The codec name (e.g., "h264", "av1"), or `None` if the codec
        cannot be determined due to a missing stream, invalid file, or execution error.

    Raises:
        None. All errors are handled internally, and `None` is returned on failure.

    Example:
        >>> _ffprobe_codec("/videos/sample.mp4")
        'h264'
        >>> _ffprobe_codec("/videos/broken_file.mp4")
        None
    """
    # Construct ffprobe command to extract the codec name from the first video stream
    cmd = [
        "ffprobe", "-v", "error",               # Suppress all non-error logs
        "-select_streams", "v:0",               # Select the first video stream only
        "-show_entries", "stream=codec_name",   # Request only the codec_name field
        "-of", "default=nw=1:nk=1",             # Simplify output (no headers/keys)
        video_path                              # Target video file
    ]

    try:
        # Execute the ffprobe command and capture the codec name as plain text
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()

        # Return codec name if non-empty, else None
        return out or None

    except Exception:
        # Return None on any error (e.g., ffprobe not found, invalid file)
        return None


def _reencode_to_h264_sw(video_path: str) -> Optional[str]:
    """
    Re-encode a video to H.264 (video) and AAC (audio) using software decoding.

    This function converts potentially problematic video codecs—such as AV1—into
    a widely compatible format (H.264/AAC). It explicitly disables hardware
    acceleration to prevent decode errors on systems with limited AV1 support.
    The re-encoded file is saved alongside the input file, with the suffix
    `_reencoded.mp4`.

    Args:
        video_path (str): Path to the input video file to be re-encoded.

    Returns:
        Optional[str]: Path to the successfully re-encoded video file, or `None`
        if re-encoding fails or output file creation fails.

    Raises:
        None. All errors are handled internally and logged.

    Example:
        >>> _reencode_to_h264_sw("/videos/input_av1.mp4")
        '/videos/input_av1_reencoded.mp4'
        >>> _reencode_to_h264_sw("/videos/missing_file.mp4")
        None
    """
    # Derive output filename (e.g., "input.mp4" -> "input_reencoded.mp4")
    base, _ = os.path.splitext(video_path)
    out_path = base + "_reencoded.mp4"

    # If a previously re-encoded version exists, reuse it to save processing time
    if os.path.exists(out_path):
        return out_path

    # FFmpeg command configuration:
    # - Disable hardware acceleration to ensure software-based decoding (more robust with AV1)
    # - Convert video to H.264 (libx264) and audio to AAC
    # - Use standard pixel format (yuv420p) and fast encoding preset
    # - CRF=20 provides good balance between quality and size
    # - "+faststart" allows progressive streaming compatibility
    cmd = [
        "ffmpeg", "-y",               # Overwrite output file if it exists
        "-hwaccel", "none",           # Force software decoding
        "-i", video_path,             # Input video
        "-map", "0:v:0",              # Select first video stream
        "-c:v", "libx264",            # Encode video using H.264 codec
        "-pix_fmt", "yuv420p",        # Standard pixel format for compatibility
        "-preset", "veryfast",        # Optimize for speed over compression
        "-crf", "20",                 # Quality factor (lower = higher quality)
        "-map", "0:a?",               # Include first audio stream if available
        "-c:a", "aac",                # Encode audio using AAC codec
        "-movflags", "+faststart",    # Enable fast start for web playback
        out_path                      # Output file path
    ]

    try:
        # Run the FFmpeg command, suppressing output and raising an exception on failure
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Return output path only if the file was created successfully
        return out_path if os.path.exists(out_path) else None

    except Exception as e:
        # Log the failure with details for debugging
        logger.error("FFmpeg software re-encode failed for {}: {}", video_path, e)
        return None


def safe_video_capture(video_path: str) -> Optional[cv2.VideoCapture]:
    """
    Open a video file safely and robustly, handling problematic codecs like AV1.

    This function attempts to create a `cv2.VideoCapture` object for a given video path.
    If the video uses the AV1 codec (which can cause hardware decoding issues on some
    systems), it first re-encodes the video to H.264 using software decoding via FFmpeg.
    If direct opening fails for other codecs, it performs the same fallback re-encoding.

    This ensures compatibility across a wide range of platforms and OpenCV builds that
    may not natively support AV1 or certain exotic codecs.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        Optional[cv2.VideoCapture]: 
            A valid OpenCV `VideoCapture` object if the video was successfully opened,
            or `None` if all attempts to open the video (original and re-encoded) fail.

    Raises:
        None. All errors are handled internally and logged.

    Example:
        >>> cap = safe_video_capture("/videos/sample_av1.mp4")
        >>> if cap and cap.isOpened():
        ...     print("Video successfully opened!")
        ... else:
        ...     print("Failed to open video.")
    """
    # Step 1: Check codec type using ffprobe to detect AV1 sources
    codec = _ffprobe_codec(video_path)
    if codec and codec.lower() in {"av1"}:
        # Attempt software re-encode to H.264 for AV1 sources
        reenc = _reencode_to_h264_sw(video_path)
        if reenc:
            cap = cv2.VideoCapture(reenc)
            if cap.isOpened():
                logger.info("Opened AV1 source via software reencode: {}", reenc)
                return cap

    # Step 2: Attempt to open video directly with OpenCV
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap

    # Step 3: Fallback — re-encode even if codec wasn’t explicitly detected as AV1.
    # This covers edge cases like corrupted headers or unusual codecs.
    reenc = _reencode_to_h264_sw(video_path)
    if reenc:
        cap2 = cv2.VideoCapture(reenc)
        if cap2.isOpened():
            logger.info("Opened source via fallback reencode: {}", reenc)
            return cap2

    # Step 4: If all methods fail, log an error and return None
    logger.error("Could not open video (original or reencoded): {}", video_path)
    return None


def save_frames_with_mapping(video_path: str, frame_numbers: List[int], save_dir: str,
                             video_mapping: Dict[str, Tuple[str, str, int]]) -> None:
    """
    Extract and save specific frames from a video, naming them according to city, country, and video ID.

    This function reads a video file, extracts frames at specific frame indices,
    and saves them as image files in the specified directory. Each frame file
    is named following the convention:
    `{city}_{country}_{videoid}_{frame_number}.jpg`.

    The function ensures:
    - Robust video opening via `safe_video_capture()` (handles AV1 or corrupt videos).
    - Frame validity (skips invalid or unreadable frames).
    - Logging for all major operations and errors.

    Args:
        video_path (str): Path to the input video file.
        frame_numbers (List[int]): List of absolute frame indices to extract and save.
        save_dir (str): Directory where extracted frames will be stored.
        video_mapping (Dict[str, Tuple[str, str]]): Dictionary mapping video IDs
            to their corresponding (city, country) metadata, usually from `mapping.csv`.

    Returns:
        None. Frames are saved as image files. Errors and warnings are logged.

    Raises:
        None. All errors are handled internally.

    Example:
        >>> mapping = {"vid001": ("Paris", "France")}
        >>> save_frames_with_mapping(
        ...     "/videos/vid001.mp4",
        ...     [100, 500, 1000],
        ...     "/output/frames",
        ...     mapping,
        ... )
        # Output files:
        # /output/frames/Paris_France_vid001_100.jpg
        # /output/frames/Paris_France_vid001_500.jpg
        # /output/frames/Paris_France_vid001_1000.jpg
    """
    # Ensure the output directory exists
    
    os.makedirs(save_dir, exist_ok=True)

    # Attempt to open the video file safely (handles AV1 or codec issues)
    cap = safe_video_capture(video_path)
    if cap is None or not cap.isOpened():
        logger.error("Failed to open video file: {}", video_path)
        return

    # Extract video ID from filename (e.g., "videos/vid123.mp4" -> "vid123")
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate that the video ID exists in the provided mapping
    if video_id not in video_mapping:
        logger.error(
            "Video ID '{}' not found in mapping.csv. Frames will be skipped for this video.",
            video_id,
        )
        cap.release()
        return

    # Retrieve the (city, country) pair for naming output images
    city, country, time_of_day = video_mapping[video_id]

    time_label = "day" if time_of_day == 0 else "night"

    # Create subdirectory (e.g., /output/frames/day/)
    subdir = os.path.join(save_dir, time_label)
    os.makedirs(subdir, exist_ok=True)

    # Iterate through requested frame indices
    for frame_num in frame_numbers:
        # Skip invalid frame indices (negative or beyond video length)
        if frame_num < 0 or frame_num >= total_frames:
            continue

        # Seek to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame {} from {}", frame_num, video_path)
            continue

        # Build output filename: city_country_videoid_framenum.jpg
        filename = f"{city}_{country}_{video_id}_{frame_num}.jpg"
        out_path = os.path.join(subdir, filename)

        # Save frame as a JPEG image
        cv2.imwrite(out_path, frame)
        logger.info("Saved frame {} -> {}", frame_num, out_path)

    # Release video resources after all frames are processed
    cap.release()


def main() -> None:
    """
    Run the mapping-first frame extraction workflow.

    This entry point orchestrates the full pipeline:

    1) Load configuration values (paths, thresholds, limits).
    2) Load the video-to-(city,country) mapping from ``mapping.csv``.
    3) Iterate the rows of ``mapping.csv`` in file order.
       For each row with a non-empty ``country``:
         a) For each ``video_id`` listed in the row:
            i.   Find that video's YOLO CSV files in ``BBOX_DIR``.
            ii.  Select valid absolute frame indices (meeting thresholds and
                 sustained for a configured rolling window).
            iii. Locate the corresponding ``.mp4`` in any configured ``video_dirs``.
            iv.  Extract and save frames to ``SAVE_DIR`` as
                 ``{city}_{country}_{videoid}_{frame}.jpg``.

    Behavior:
      * If a required config is missing, the function logs an error and exits.
      * If the mapping file is empty/unreadable, the function logs an error and exits.
      * If ``STOP_AFTER_FIRST_VALID_MAPPING_ROW`` is True, the loop stops after
        processing the first row that has a non-empty country.

    Args:
        None.

    Returns:
        None. Side effects include reading/writing files and logging progress/errors.

    Raises:
        None. All errors are handled and logged; the function returns early on failure.

    Example:
        Run as a script:

            if __name__ == "__main__":
                main()
    """
    try:
        # ---- Load required configuration ----
        bbox_dir = common.get_configs("BBOX_DIR") or ""
        video_dirs = common.get_configs("video_dirs") or []
        save_dir: str = common.get_configs("SAVE_DIR") or "./frames"

        min_persons: int = common.get_configs("MIN_PERSONS") or 0
        min_cars: int = common.get_configs("MIN_CARS") or 0
        min_lights: int = common.get_configs("MIN_LIGHTS") or 0
        max_frames: int = common.get_configs("MAX_FRAMES") or 100
        window: int = common.get_configs("CONF_WINDOW") or 10

        # Path to the mapping file that ties video IDs to (city, country)
        mapping_csv_path = common.get_configs("mapping")

    except KeyError as e:
        # A specific, required configuration key is missing.
        logger.error("Missing required configuration key: {}", e)
        return
    except Exception as e:
        # Any other configuration-loading failure.
        logger.error("Configuration loading failed. Error: {}", e)
        return

    # ---- Build lookup: video_id -> (city, country) ----
    video_mapping = get_video_mapping(mapping_csv_path)
    if not video_mapping:
        logger.error("Empty or unreadable mapping.csv; nothing to do.")
        return

    processed_any_row = False

    try:
        # Read mapping rows in file order
        with open(mapping_csv_path, newline="", encoding="utf-8") as csvfile:  # type: ignore
            reader = csv.DictReader(csvfile)

            for row in reader:
                city = row.get("city", "")
                country = row.get("country", "")
                videos_str = row.get("videos", "")

                # Skip rows without a country; we need (city, country) in the filename.
                if not country:
                    logger.warning("Row for city='{}' has empty country; skipping.", city)
                    continue

                # Parse the row's list of video IDs (robust to different formats).
                videos_list = parse_videos_list_field(videos_str)

                logger.info(
                    "Processing mapping row: city='{}', country='{}', videos={}",
                    city,
                    country,
                    len(videos_list),
                )

                # ---- Process each video listed in this mapping row ----
                for video_id in videos_list:
                    # 1) Find the YOLO CSVs for this video.
                    csv_paths = glob_csvs_for_video(bbox_dir, video_id)
                    if not csv_paths:
                        logger.warning("No CSVs for video_id='{}'; skipping.", video_id)
                        continue

                    # 2) From those CSVs, select absolute frame numbers that meet
                    #    detection thresholds over a stable rolling window.
                    frame_numbers = select_frames_for_csvs(
                        csv_paths, min_persons, min_cars, min_lights, max_frames, window
                    )
                    if not frame_numbers:
                        logger.info("No frames matched thresholds for video_id='{}'.", video_id)
                        continue

                    # 3) Locate the concrete .mp4 file across configured directories.
                    found_video: Optional[str] = None
                    for folder in video_dirs:
                        candidate = os.path.join(folder, f"{video_id}.mp4")
                        if os.path.exists(candidate):
                            found_video = candidate
                            break

                    if not found_video:
                        logger.error("Video file for ID '{}' not found in any directory.", video_id)
                        continue

                    # 4) Extract and save frames using the mapping for filename metadata.
                    save_frames_with_mapping(found_video, frame_numbers, save_dir, video_mapping)

                processed_any_row = True

                # Optionally stop after the first valid row, if configured.
                if STOP_AFTER_FIRST_VALID_MAPPING_ROW:
                    logger.info("STOP_AFTER_FIRST_VALID_MAPPING_ROW=True — stopping after this row.")
                    break

        # If we iterated the file but never found a row with a country, inform the user.
        if not processed_any_row:
            logger.warning("No mapping rows with a non-empty 'country' were processed.")

    except Exception as e:
        # Any unexpected runtime error while processing the mapping file.
        logger.error("Failed while iterating mapping.csv rows. Error: {}", e)


if __name__ == "__main__":
    main()
