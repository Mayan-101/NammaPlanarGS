import os
import subprocess
from pathlib import Path

# 1. Resolve absolute paths
BASE_DIR = Path("/marimo/NammaPlanarGS/preprocess").resolve()
images_dir = BASE_DIR / "images"

# If the images aren't in that exact directory, check current working dir
if not images_dir.is_dir():
    images_dir = Path("images").resolve()

# Verify images actually exist
image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
print(f"Image directory: {images_dir}")
print(f"Found {len(image_files)} frames.")
if len(image_files) == 0:
    raise RuntimeError(f"No .jpg or .png images found in {images_dir}")

colmap_out = BASE_DIR / "colmap_output"
database_path = colmap_out / "database.db"
sparse_dir = colmap_out / "sparse"

colmap_out.mkdir(parents=True, exist_ok=True)
sparse_dir.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# 2. Feature Extraction (Populates database.db with keypoints & camera intrinsics)
print("\n--- Step 1: Feature Extraction ---")
subprocess.run(
    [
        "colmap",
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
    ],
    env=env,
    check=True,
)

# 3. Feature Matching (Sequential matcher for continuous video frames)
print("\n--- Step 2: Feature Matching ---")
subprocess.run(
    [
        "colmap",
        "sequential_matcher",
        "--database_path", str(database_path),
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

print("\nReconstruction completed successfully!")