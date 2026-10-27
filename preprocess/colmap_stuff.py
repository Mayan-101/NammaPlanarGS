import os
import shutil
import subprocess
import sys


def run_command(cmd: list[str], desc: str):
    """Runs a shell command and streams output, exiting on error."""
    print(f"\n{'='*20} {desc} {'='*20}")
    print(f"Executing: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{desc}' failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def run_colmap_sparse(
    image_dir: str = "frames",
    workspace_dir: str = "colmap_workspace",
    camera_model: str = "OPENCV",
    matcher_type: str = "exhaustive",  # "exhaustive" or "sequential"
    use_gpu: bool = True,
):
    """
    Runs the complete COLMAP sparse reconstruction pipeline:
      1. Feature Extraction
      2. Feature Matching (Exhaustive or Sequential)
      3. Sparse Reconstruction (Mapper)
    """
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Prepare directories and database path
    os.makedirs(workspace_dir, exist_ok=True)
    database_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"

    # -------------------------------------------------------------
    # 1. Feature Extraction
    # -------------------------------------------------------------
    # Single camera is enabled by default since frames originate from the same video.
    extract_cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu_flag,
    ]
    run_command(extract_cmd, "Step 1: Feature Extraction")

    # -------------------------------------------------------------
    # 2. Feature Matching
    # -------------------------------------------------------------
    if matcher_type == "sequential":
        # Best for continuous video sequences (> 500 frames)
        match_cmd = [
            "colmap", "sequential_matcher",
            "--database_path", database_path,
            "--SequentialMatching.overlap", "20",
            "--SequentialMatching.loop_detection", "1",
            "--SiftMatching.use_gpu", gpu_flag,
        ]
    else:
        # Best for high accuracy when total frames < 500
        match_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
            "--SiftMatching.use_gpu", gpu_flag,
        ]
    run_command(match_cmd, f"Step 2: Feature Matching ({matcher_type})")

    # -------------------------------------------------------------
    # 3. Sparse Reconstruction (Mapping)
    # -------------------------------------------------------------
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
    ]
    run_command(mapper_cmd, "Step 3: Sparse Reconstruction (Mapper)")

    # -------------------------------------------------------------
    # Verification of Output
    # -------------------------------------------------------------
    # COLMAP places the main reconstructed model in `sparse/0/`
    model_dir = os.path.join(sparse_dir, "0")
    if os.path.isdir(model_dir):
        print(f"\nSparse reconstruction finished successfully!")
        print(f"Model saved to: {model_dir}")
        print(f"Generated files: {', '.join(os.listdir(model_dir))}")
    else:
        print(
            "\n[Warning] No model found in 'sparse/0/'. "
            "Reconstruction may have failed to find enough visual overlap.",
            file=sys.stderr
        )


if __name__ == "__main__":
    run_colmap_sparse(
        image_dir="images",
        workspace_dir="colmap_output",
        camera_model="OPENCV",      # Standard for distorted pinhole / phone cameras
        matcher_type="sequential",   # Use "sequential" if you have a large continuous video (> 500 frames)
        use_gpu=True,                # Set to False if running on a CPU-only environment
    )