import os
import shutil
import subprocess
from pathlib import Path

# 1. Resolve paths
BASE_DIR = Path("/marimo/NammaPlanarGS/preprocess").resolve()
images_dir = BASE_DIR / "images"

if not images_dir.is_dir():
    images_dir = Path("images").resolve()

image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
print(f"Image directory: {images_dir}")
print(f"Found {len(image_files)} frames.")
if len(image_files) == 0:
    raise RuntimeError(f"No .jpg or .png images found in {images_dir}")

colmap_out = BASE_DIR / "colmap_output"
database_path = colmap_out / "database.db"
sparse_dir = colmap_out / "sparse"
undistorted_dir = colmap_out / "undistorted"

colmap_out.mkdir(parents=True, exist_ok=True)
sparse_dir.mkdir(parents=True, exist_ok=True)
undistorted_dir.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# 2. Feature Extraction
print("\n--- Step 1: Feature Extraction ---")
subprocess.run(
    [
        "colmap",
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ],
    env=env,
    check=True,
)

# 3. Sequential Matching with Loop Detection
print("\n--- Step 2: Feature Matching ---")
subprocess.run(
    [
        "colmap",
        "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "15",
        "--SequentialMatching.loop_detection", "1",
        "--SiftMatching.use_gpu", "1",
    ],
    env=env,
    check=True,
)

# 4. Sparse Reconstruction (Mapper)
print("\n--- Step 3: Sparse Reconstruction ---")
subprocess.run(
    [
        "colmap",
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ],
    env=env,
    check=True,
)

# Verify reconstruction produced a valid primary sub-model
primary_model_dir = sparse_dir / "0"
if not primary_model_dir.exists():
    raise RuntimeError(
        f"Reconstruction failed to create model '0' in {sparse_dir}. "
        "Check feature matches or frame overlap."
    )

# 5. Image Undistortion (Required for Gaussian Splatting)
print("\n--- Step 4: Image Undistortion for Gaussian Splatting ---")
subprocess.run(
    [
        "colmap",
        "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(primary_model_dir),
        "--output_path", str(undistorted_dir),
        "--output_type", "COLMAP",
        "--max_image_size", "2000",
    ],
    env=env,
    check=True,
)

print("\n--- Preprocessing Complete ---")
print(f"Gaussian Splatting dataset ready at: {undistorted_dir}")
print(f"  - Undistorted Images: {undistorted_dir / 'images'}")
print(f"  - Sparse Model:       {undistorted_dir / 'sparse'}")