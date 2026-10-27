# 1. Download standalone single-file micromamba executable
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

# 2. Install pre-compiled COLMAP with CUDA into /opt/colmap-cuda
micromamba create -y -p /opt/colmap-cuda -c conda-forge "colmap=*=*cuda*"

# 3. Symlink the colmap CLI executable into /usr/local/bin
ln -sf /opt/colmap-cuda/bin/colmap /usr/local/bin/colmap

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/colmap-cuda/lib:$LD_LIBRARY_PATH

