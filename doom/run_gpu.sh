#!/usr/bin/env bash
# Run the Doom brain with the cuTile GPU backend.
# CUDA 13.2 (with tileiras) lives in /usr/local/cuda; /usr/bin/nvcc is an older
# toolkit, so the CUDA bin directory must come first on PATH. The .bin shim
# provides clang++ on hosts that only have g++. BLAS stays single-threaded for
# the same reason as doom/run_broadcast.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PWD/.bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export DOOM_BRAIN_BACKEND="${DOOM_BRAIN_BACKEND:-cutile}"
exec .venv-neural/bin/python -m doom.server "$@"
