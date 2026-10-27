import os
import re
import sys
import gdown
import ffmpeg


def extract_drive_id(drive_url_or_id: str) -> str:
    """
    Extracts the file ID if a full Google Drive URL is provided.
    Accepts full sharing URLs or a raw file ID.
    """
    # Matches /file/d/<id>/ or id=<id>
    match = re.search(r"(?:/file/d/|id=)([a-zA-Z0-9_-]+)", drive_url_or_id)
    if match:
        return match.group(1)
    # If no URL pattern matched, assume drive_url_or_id is already the raw ID
    return drive_url_or_id.strip()


def download_from_drive(drive_url_or_id: str, output_path: str = "downloaded_video.mp4") -> str:
    """
    Downloads a video file from Google Drive using gdown by file ID.
    """
    file_id = extract_drive_id(drive_url_or_id)

    print(f"Downloading video from Google Drive (ID: {file_id})...")
    
    # Passing id= directly handles confirmation prompts and avoids URL construction issues
    downloaded_file = gdown.download(id=file_id, output=output_path, quiet=False)

    if not downloaded_file or not os.path.isfile(downloaded_file):
        raise RuntimeError("Failed to download video from Google Drive.")

    print(f"Video saved to: {downloaded_file}")
    return downloaded_file

def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    fps: float | None = None,
    use_gpu: bool = False,
):
    """
    Extracts frames from the downloaded video using ffmpeg-python.
    """
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    input_kwargs = {}
    if use_gpu:
        input_kwargs["hwaccel"] = "cuda"
    else:
        input_kwargs["threads"] = 0

    stream = ffmpeg.input(video_path, **input_kwargs)

    if fps is not None:
        stream = stream.filter("fps", fps=fps, round="up")

    output_kwargs = {
        "start_number": 1,
        "qscale:v": 3,
        "pix_fmt": "yuvj420p",
        "threads": 0,
    }

    try:
        print(f"Extracting frames to '{output_dir}/'...")
        (
            stream.output(output_pattern, **output_kwargs)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Extraction complete: '{output_dir}/'")

    except ffmpeg.Error as e:
        print("FFmpeg error occurred:", file=sys.stderr)
        print(e.stderr.decode("utf8"), file=sys.stderr)


def process_drive_video(
    drive_url_or_id: str,
    video_save_path: str = "input_video.mp4",
    frames_dir: str = "frames",
    fps: float | None = None,
    use_gpu: bool = False,
    delete_video_after: bool = False,
):
    """
    Full pipeline: Downloads video from Drive and extracts frames.
    """
    # 1. Download
    video_file = download_from_drive(drive_url_or_id, video_save_path)

    # 2. Extract frames
    extract_frames(video_file, output_dir=frames_dir, fps=fps, use_gpu=use_gpu)

    # 3. Optional cleanup of the large video file
    if delete_video_after and os.path.exists(video_file):
        os.remove(video_file)
        print(f"Cleaned up temporary video file '{video_file}'.")


if __name__ == "__main__":
    # Paste your Google Drive link or File ID here
    DRIVE_LINK = "https://drive.google.com/file/d/1yHNEPL7tPRSUCjYT2rtWBRKH-8XX2W8n/view?usp=sharing"

    process_drive_video(
        drive_url_or_id=DRIVE_LINK,
        video_save_path="VID20261023160251.mp4",
        frames_dir="frames",
        fps=None,         # e.g., set to 1 for 1 frame/sec, or None for all frames
        use_gpu=True,    # Set to True if your FFmpeg has CUDA support
        delete_video_after=True  # Set to True if you want to free up disk space after extraction
    )