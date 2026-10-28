#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# 1. Install Micromamba & Isolated COLMAP-CUDA
if [ ! -f /usr/local/bin/micromamba ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
fi

if [ ! -d /opt/colmap-cuda ]; then
    micromamba create -y -p /opt/colmap-cuda -c conda-forge "colmap=*=*cuda*"
fi

# Wrap colmap so /opt/colmap-cuda/lib ONLY applies to colmap, NOT Python/PyTorch!
cat << 'EOF' > /usr/local/bin/colmap
#!/usr/bin/env bash
export LD_LIBRARY_PATH="/opt/colmap-cuda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM="offscreen"
exec /opt/colmap-cuda/bin/colmap "$@"
EOF
chmod +x /usr/local/bin/colmap

# 2. Create/Sync Python 3.10 venv using uv (matches your cpython-310 .so files!)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=$(nproc)

if [ ! -d "$REPO_ROOT/.venv" ]; then
    uv venv "$REPO_ROOT/.venv" 
fi

VENV_PY="$REPO_ROOT/.venv/bin/python"


uv pip install --python "$VENV_PY" -r requirements.txt

# 3. Clone GroundedSAM if missing (since it's in .gitignore)
if [ ! -d "$REPO_ROOT/submodules/groundedsam" ]; then
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git "$REPO_ROOT/submodules/groundedsam"
fi

# 4. Install C++/CUDA submodules with --no-build-isolation
uv pip install --python "$VENV_PY" --no-build-isolation \
    -e "$REPO_ROOT/submodules/simple-knn" \
    -e "$REPO_ROOT/submodules/pytorch3d" \
    "$REPO_ROOT/submodules/diff-plane-rasterization" \
    -e "$REPO_ROOT/submodules/groundedsam/segment_anything" \
    -e "$REPO_ROOT/submodules/groundedsam/GroundingDINO"

# Optional: Compile CroCo RoPE CUDA kernel for 2x faster DUSt3R
uv pip install --python "$VENV_PY" --no-build-isolation \
    -e "$REPO_ROOT/submodules/dust3r/croco/models/curope" || true

echo "Setup complete! Use $VENV_PY to run scripts."