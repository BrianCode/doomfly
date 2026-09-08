# Doomfly handoff: pick up on the GPU machine

State of this directory on 2026-09-08:

- Repo: https://github.com/nftechie/doomfly, shallow clone, unmodified source.
- MaleCNS v1.0 data downloaded and checksum-verified in `connectome_data/malecns_v1/`.
- Imported graph in `outputs/doom/malecns_v1/graph.npz` (166,700 neurons, 25.6 M edges, 239 MB).
- A 12-minute local training run in `outputs/doom/local-training/` (checkpoints and audit log).
- Analysis of the Feather tables in `handoff/` (stats, heatmap CSV, report page).
- `.bin/clang++` is a shim that calls `g++`. The build scripts hardcode `clang++`.

Not included: `.venv-neural` (rebuild in 1 minute), `.git`.

## 1. Unpack

```sh
tar -xzf doomfly-handoff.tar.gz
cd doomfly
```

## 2. Toolchain

Needs Python 3.11, a C++ compiler, and for the GPU work an NVIDIA driver r580 or newer.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.11
~/.local/bin/uv venv -p 3.11 .venv-neural
~/.local/bin/uv pip install -p .venv-neural/bin/python \
  -r requirements-neural.txt -r doom/requirements.txt \
  --build-constraints neural-build-constraints.txt
which clang++ || export PATH=$PWD/.bin:$PATH
```

## 3. Rebuild the native kernel and verify

The kernel binary is host-specific. Rebuild it, then run the reference tests.

```sh
.venv-neural/bin/python -m doom.build_kernel
.venv-neural/bin/python -m pytest tests/test_doom.py tests/test_doom_reference.py -q
```

Expected: 12 passed.

## 4. Run the brain

```sh
export PATH=$PWD/.bin:$PATH
OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy .venv-neural/bin/python -m doom.server \
  --model experimental-v6 --learning --port 8766 \
  --audit-dir outputs/doom/local-training \
  --checkpoint-dir outputs/doom/local-training/checkpoints \
  --checkpoint-seconds 300 --resume
curl -s localhost:8766/health
```

`--resume` continues from the checkpoint made on the first machine. Drop it for a fresh run.

## 5. Baseline before any GPU work

Record the CPU number on the new machine first:

```sh
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python - <<'PY'
import numpy as np
from doom.native import NativeBrain
b=NativeBrain('outputs/doom/malecns_v1/graph.npz')
rng=np.random.default_rng(0); lum=rng.uniform(0,1,len(b.retina)).astype(np.float32)
for _ in range(10): b.step(lum,28.6)
t=[b.step(lum,28.6)[1] for _ in range(20)]
print('kernel ms per 28.6 ms tic: %.1f'%(1e3*np.mean(t)))
PY
```

First machine (Intel Core Ultra 7 155H, one core): 45 ms per tic.
GPU machine (Intel Core i7-13620H, one core, 2026-09-08): 28.0 ms per tic (min 26.7, max 30.9).

## 6. GPU checks

```sh
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
```

RTX 3060 must show compute_cap 8.6 and driver >= 580 for cuTile. Then:

```sh
~/.local/bin/uv pip install -p .venv-neural/bin/python cuda-tile==1.5.0 cupy-cuda13x==13.6.0
```

Keep `cupy-cuda13x` at 13.6.0: CuPy 14 requires numpy 2, which breaks the pinned numpy 1.24.4
that numba and Brian2 need. `/usr/local/cuda` must be CUDA 13.2 (with `bin/tileiras`); an older
`/usr/bin/nvcc` is fine as long as `/usr/local/cuda/bin` comes first on PATH (see `doom/run_gpu.sh`).
CuPy needs `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.

Port order, smallest risk first:

1. `doom/kernel.cpp` LIF evolve + spike delivery + reset. Evolve all 166,700 neurons per substep on the GPU; keep the 18-substep delay batching in mind.
2. Retina luminance transform in `doom/engine.py` (`retinal_samples`).
3. `doom_learning_v6/kernel.cpp` plasticity rule, 4,184 edges.

Validate every port against `tests/test_doom_reference.py`, which is the Brian2 oracle.

## Graph arrays for a GPU port

`outputs/doom/malecns_v1/graph.npz` keys: `ptr` int64[166701], `post` int32[25.6M],
`weight` float32[25.6M] (signed mV), `ids`, `retina`, `lamina`, `sugar`, `uv`, `superclass`.
Model constants: tau_m 20 ms, tau_g 5 ms, rest -52 mV, threshold -45 mV,
refractory 2.2 ms, delay 1.8 ms, dt 0.1 ms.

## 7. GPU backend and video export (added 2026-09-08 on the fork)

```sh
# relabel the checkpoint identity for the current source tree and backend, then resume on the GPU
.venv-neural/bin/python -m doom.convert_checkpoint --checkpoint-dir outputs/doom/local-training/checkpoints --model experimental-v6 --backend cutile
bash doom/run_gpu.sh --model experimental-v6 --learning --backend cutile --port 8766 \
  --audit-dir outputs/doom/local-training --checkpoint-dir outputs/doom/local-training/checkpoints \
  --checkpoint-seconds 300 --resume --video-mp4 outputs/doom/video
curl -s localhost:8766/video/health
```

`doom/run_gpu.sh` sets the CUDA 13.2 paths and `DOOM_BRAIN_BACKEND=cutile`. The checkpoint
identity hashes every `doom/*.py`, so run the converter after any source change before `--resume`
(it refuses if the model configuration differs). Video: `--video-mp4 DIR` writes
`DIR/<run_id>/archive-%05d.mp4` (35 f/s = 1x brain time, 5-minute fragmented segments),
`video-metrics.jsonl` (one row per frame, joinable to `audit.jsonl` on run_id + tick) and
`video-manifest.json`. `--video-encoder auto` picks the first working of vaapi (Intel iGPU),
qsv, nvenc, x264. Afterwards: `python -m doom.video_charts --run DIR/<run_id> --movie` and
`python -m doom.video_clips --run DIR/<run_id> --events damage,kill,round_end`.
Extra packages: `requirements-gpu.txt`. Optimization notes: `docs/gpu-optimization-tracker.md`.

GPU speed on this laptop is limited by its power policy (memory clock held at 810 MHz, 25 W cap);
`sudo nvidia-smi -pl 35` / `sudo nvidia-smi -lmc 5001,5501` or installing `nvidia-powerd` may lift it.
QSV needs `sudo apt install libmfx-gen1.2`; VAAPI works without it.
