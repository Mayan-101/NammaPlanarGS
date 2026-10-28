import os
import re
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
import gdown
import ffmpeg


def extract_drive_id(drive_url_or_id: str) -> str:
    """
    Extracts the file ID if a full Google Drive URL is provided.
    Accepts full sharing URLs or a raw file ID.
    """
    match = re.search(r"(?:/file/d/|id=)([a-zA-Z0-9_-]+)", drive_url_or_id)
    if match:
        return match.group(1)
    return drive_url_or_id.strip()


def download_from_drive(drive_url_or_id: str, output_path: str = "downloaded_video.mp4") -> str:
    """
    Downloads a video file from Google Drive using gdown by file ID.
    """
    file_id = extract_drive_id(drive_url_or_id)
    print(f"Downloading video from Google Drive (ID: {file_id})...")
    downloaded_file = gdown.download(id=file_id, output=output_path, quiet=False)

    if not downloaded_file or not os.path.isfile(downloaded_file):
        raise RuntimeError("Failed to download video from Google Drive.")

    print(f"Video saved to: {downloaded_file}")
    return downloaded_file


def compute_tile_sharpness(gray_img: np.ndarray, grid_size: int = 4, top_k_ratio: float = 0.25) -> float:
    """
    Computes sharpness using the top-K highest-gradient tiles so that large
    textureless indoor walls do not artificially depress the sharpness score.
    """
    h, w = gray_img.shape
    tile_h, tile_w = h // grid_size, w // grid_size
    scores = []
    for r in range(grid_size):
        for c in range(grid_size):
            tile = gray_img[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w]
            lap_var = cv2.Laplacian(tile, cv2.CV_64F).var()
            scores.append(lap_var)
    scores.sort(reverse=True)
    k = max(1, int(len(scores) * top_k_ratio))
    return float(np.mean(scores[:k]))


def select_sharp_keyframes(
    raw_frames_dir: str,
    output_dir: str,
    target_keyframes: int = 120,
    min_flow_ratio: float = 0.04,
):
    """
    Two-stage keyframe filter tailored for PlanarGS:
    Stage 1: Enforces minimum optical flow / parallax baseline (ignores pure stationary pauses).
    Stage 2: Within each temporal window, selects the sharpest frame using tile-based Laplacian variance.
    """
    raw_paths = sorted(Path(raw_frames_dir).glob("*.jpg"))
    if not raw_paths:
        raise RuntimeError(f"No candidate frames found in {raw_frames_dir}")

    print(f"Scoring {len(raw_paths)} candidate frames for sharpness and parallax...")

    # Stage 1: Filter out stationary / redundant frames using sparse Lucas-Kanade optical flow
    motion_filtered = []
    prev_gray = None
    prev_pts = None

    for p in raw_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = compute_tile_sharpness(gray)

        if prev_gray is None:
            motion_filtered.append((p, sharpness))
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=15)
            continue

        if prev_pts is not None and len(prev_pts) >= 15:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            valid_prev = prev_pts[status.flatten() == 1]
            valid_next = next_pts[status.flatten() == 1]
            if len(valid_next) >= 10:
                displacements = np.linalg.norm(valid_next - valid_prev, axis=-1)
                median_disp = float(np.median(displacements)) / float(max(gray.shape))
                # Skip frame if camera barely moved
                if median_disp < min_flow_ratio:
                    continue

        motion_filtered.append((p, sharpness))
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=15)

    print(f"Retained {len(motion_filtered)} frames after motion/parallax filtering.")

    # Stage 2: Sliding-window local sharpness maxima selection to hit target_keyframes
    if len(motion_filtered) <= target_keyframes:
        selected = [item[0] for item in motion_filtered]
    else:
        chunks = np.array_split(np.arange(len(motion_filtered)), target_keyframes)
        selected = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            best_idx = max(chunk, key=lambda idx: motion_filtered[idx][1])
            selected.append(motion_filtered[best_idx][0])

    os.makedirs(output_dir, exist_ok=True)
    # Clear existing frames in output_dir
    for existing in Path(output_dir).glob("*.jpg"):
        existing.unlink()

    for idx, src_path in enumerate(selected, 1):
        dst_path = Path(output_dir) / f"frame_{idx:04d}.jpg"
        shutil.copy2(src_path, dst_path)

    print(f"Selected {len(selected)} sharp keyframes saved to '{output_dir}/'.")


def extract_frames(
    video_path: str,
    output_dir: str = "images",
    candidate_fps: float = 6.0,
    target_keyframes: int = 120,
    min_flow_ratio: float = 0.04,
    use_gpu: bool = False,
):
    """
    Extracts candidate frames via FFmpeg and filters them down to the sharpest,
    well-spaced keyframes for COLMAP and PlanarGS.
    """
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.", file=sys.stderr)
        return

    raw_candidates_dir = f"{output_dir}_raw_candidates"
    os.makedirs(raw_candidates_dir, exist_ok=True)
    output_pattern = os.path.join(raw_candidates_dir, "cand_%05d.jpg")

    input_kwargs = {}
    if use_gpu:
        input_kwargs["hwaccel"] = "cuda"
    else:
        input_kwargs["threads"] = 0

    stream = ffmpeg.input(video_path, **input_kwargs)
    if candidate_fps is not None and candidate_fps > 0:
        stream = stream.filter("fps", fps=candidate_fps, round="up")

    output_kwargs = {
        "start_number": 1,
        "qscale:v": 2,
        "pix_fmt": "yuvj420p",
        "threads": 0,
    }

    try:
        print(f"Extracting candidate frames at {candidate_fps} FPS to '{raw_candidates_dir}/'...")
        (
            stream.output(output_pattern, **output_kwargs)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg GPU/filter extraction failed, retrying on CPU...", file=sys.stderr)
        stream = ffmpeg.input(video_path, threads=0)
        if candidate_fps is not None and candidate_fps > 0:
            stream = stream.filter("fps", fps=candidate_fps, round="up")
        (
            stream.output(output_pattern, **output_kwargs)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

    select_sharp_keyframes(
        raw_frames_dir=raw_candidates_dir,
        output_dir=output_dir,
        target_keyframes=target_keyframes,
        min_flow_ratio=min_flow_ratio,
    )
    shutil.rmtree(raw_candidates_dir, ignore_errors=True)


def process_drive_video(
    drive_url_or_id: str,
    video_save_path: str = "input_video.mp4",
    frames_dir: str = "images",
    candidate_fps: float = 6.0,
    target_keyframes: int = 120,
    min_flow_ratio: float = 0.04,
    use_gpu: bool = True,
    delete_video_after: bool = True,
):
    video_file = download_from_drive(drive_url_or_id, video_save_path)
    extract_frames(
        video_file,
        output_dir=frames_dir,
        candidate_fps=candidate_fps,
        target_keyframes=target_keyframes,
        min_flow_ratio=min_flow_ratio,
        use_gpu=use_gpu,
    )
    if delete_video_after and os.path.exists(video_file):
        os.remove(video_file)
        print(f"Cleaned up temporary video file '{video_file}'.")


if __name__ == "__main__":
    DRIVE_LINK = "https://drive.google.com/file/d/1yHNEPL7tPRSUCjYT2rtWBRKH-8XX2W8n/view?usp=sharing"

    process_drive_video(
        drive_url_or_id=DRIVE_LINK,
        video_save_path="VID20261023160251.mp4",
        frames_dir="images",
        candidate_fps=6.0,       # Sample 6 candidates/sec, then filter by motion + sharpness
        target_keyframes=120,    # Ideal count for COLMAP + PlanarGS (3 groups of 40 in DUSt3R)
        min_flow_ratio=0.04,     # Minimum 4% median optical flow displacement between keyframes
        use_gpu=True,
        delete_video_after=True,
    )
