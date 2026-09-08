# GPU port: optimization tracker

Living record of every optimization considered for the cuTile port of the neural
kernel (`doom/cutile_brain.py`) and the video export path, with expected and
measured impact and how the options interact. Update the "Measured" column
whenever a milestone lands. Numbers are from the RTX 3050 6 GB Laptop GPU
(sm_86, driver 595.84, CUDA 13.2, cuda-tile 1.5.0, cupy 13.6.0) on the
i7-13620H host unless stated otherwise.

## Baselines

| Date | Configuration | Result |
|---|---|---|
| 2026-09-08 | CPU kernel (`doom/kernel.cpp`), one core, `OPENBLAS_NUM_THREADS=1`, HANDOFF section 5 script | 28.0 ms per 28.6 ms tic (min 26.7, max 30.9) |
| 2026-09-08 | cuTile milestone 1 (`doom.compare_backends --tics 100`), GPU power-capped (see below) | GPU 43.2 ms/tic median (mean 50.3 incl. JIT), CPU 31.8 ms/tic median in the same run; speedup 0.65x |
| 2026-09-08 | Agreement over 100 tics, identical inputs | first differing tic 2; 98.7-99.0% of neurons have identical per-tic counts; total spikes ratio 1.00012; all per-superclass totals within 0.3%; two GPU runs were bit-identical for all 100 tics |
| 2026-09-08 | Brian2 oracle (`tests/test_doom_reference.py`) on the cutile backend | passes at 0.1 ms and 10 ms cadence (exact spike bins, v and g within 0.002 mV) |

## Host power state (blocks all GPU speedups until fixed)

Measured 2026-09-08 under a sustained CuPy copy load: pstate P5, SM 1762 MHz,
**memory clock 810 MHz of 5501 MHz max**, power 22 W against a current limit of
25 W (default 35 W, max 65 W), throttle reasons `SW Power Cap` and
`SW Thermal Slowdown` active at 51 C, PCIe link gen 1 x8. Achieved bandwidth:
6-14 GB/s for plain CuPy copies (device peak is about 190 GB/s); a raw CUDA
kernel doing the same work as the cuTile evolve kernel was equally slow, so the
kernels are not the cause. `nvidia-smi -lmc` / `-pl` need root; `nvidia-powerd`
is not installed. Every kernel figure below was measured in this state and
should scale by roughly the memory clock ratio once the cap is lifted.

## Kernel time budget (milestone 1, power-capped)

| Component | Measured | Note |
|---|---|---|
| K1 `evolve_batch`, 0 substeps (state load/store only) | 0.40 ms | fixed per launch; about 6-13 MB of traffic at ~14 GB/s |
| K1 per substep | 88 us | ring row read + zero (1.3 MB); compute is not measurable |
| K1 18 substeps | 2.0-2.2 ms | independent of TN in {128, 256, 512, 1024, 2048} |
| K2 `deliver`, ~1,550 spikes (0.24 M edges) | 0.39 ms | 16 launches/tic |
| K2 empty batch (2048 CTAs) | 8 us | |
| K2 median in the 100-tic run | 1.04 ms (mean 1.76) | ~3,200 spikes per batch at 51 k spikes/tic |
| Host cost per `ct.launch` | 12-18 us | negligible |
| Drive H2D / counts D2H, pageable | 0.41 / 0.55 ms | replaced by pinned buffers (not yet re-measured) |
| Per tic (16 batches) | ~40-85 ms depending on spike rate | CPU: 28-32 ms |

## Neural kernel options

| # | Option | Expected | Measured | Risk | Numerics | Interactions | Status |
|---|---|---|---|---|---|---|---|
| 1 | Delay batching (<=18 substeps per K1 launch) | 572 -> ~32 launches/tic; state in registers | implemented; per-substep cost is the ring access only | Low | No | Base for all | M1 done |
| 2 | Refractory bound (<=1 spike/neuron/batch) => fixed spike list, no overflow | enables 1, 3 | implemented | None | No | - | M1 done |
| 3 | Spike compaction once per batch (atomic counter + cumsum + scatter after the substep loop) | saves a scan per substep | per-substep compaction cost was not measurable either way | Low | No | - | M1 done |
| 4 | int32 ptr, int32 masks | trivial | implemented | None | No | - | M1 done |
| 5 | Host sync per batch for K2 grid | ~16 syncs/tic | not used | None | No | replaced by 6 | dropped |
| 6 | Sync-free delivery (fixed grid, device-read spike count, grid-stride loop) | -1 ms/tic; prerequisite for 12 | implemented from the start; empty launch 8 us | Medium | No | enables 12 | M1 done |
| 7 | JIT warm-up at construction (zero-length batch) | removes first-tic stall | implemented | None | No | - | M1 done |
| 8 | Lazy host mirror (D2H v,g,ref only on access/checkpoint) | ~1 ms/tic in v6 (3 calls per tic) | - | Low | No | `server.py:161` reads `brain.v` | M2 |
| 9 | Pinned host buffers + async copies | 0.1-0.2 ms/tic; pageable copies measured 0.4-0.55 ms each | implemented, re-measure | Low | No | complements 22 | M1 done |
| 10 | Counts stay on device across v6 bins | ~0.2 ms/tic | - | Low | No | - | M4 |
| 11 | Tile/occupancy tuning (TN, TE, grid, hints) | 10-30% kernel time | TN had no effect while bandwidth-starved; retest after the power fix | Low | No | - | M4 |
| 12 | CUDA graph / stream capture per distinct `steps` | 0.3-0.5 ms/tic | - | Medium | No | requires 6 | M4 |
| 13 | Edge-balanced delivery | cuts the 11k-degree tail | - | Medium | No | complements 14 | M4 |
| 14 | CSR locality: sort `post` within rows; neuron permutation for atomic locality | 10-30% of K2 | - | Medium | rounding order only | complements 13 | M4 |
| 15 | Integer contact accumulation (deterministic) | deterministic run-to-run | note: two runs were already bit-identical over 100 tics; still not guaranteed | Medium | yes (documented) | alternative to 16; conflicts 18 | optional |
| 16 | Deterministic sort + segmented sum | determinism at 0.5-1 ms/batch | - | Low | yes | superseded by 15 | not recommended |
| 17 | Weight code uint16 + LUT | halves weight reads | - | Medium | no | conflicts plastic scatter | not now |
| 18 | fp16 weights | bandwidth only | - | Low | yes | - | not recommended |
| 19 | Sleeping-neuron / sleeping-tile skip | ~none (ring traffic dominates) | - | Medium | no | - | not recommended |
| 20 | Batch > 18 via refractory | impossible (delay-limited) | - | - | - | - | rejected |
| 21 | Pull/CSC delivery | no benefit vs atomics | - | High | rounding | - | rejected |
| 22 | On-device retina transform, low-pass, drive assembly | 0.5-1 ms CPU per bin; removes drive H2D | - | Medium | tiny float diffs (hash of `light` changes) | complements 9 | M4 |
| 23 | Plasticity rule on device | none (4,184 elements) | - | - | no | - | rejected |
| 24 | Move 10 Hz JPEG/JSON publish off the sim thread | 5-10 ms per publish | - | Low | no | independent of GPU; the video service already moves compositing off-thread | later |
| 25 | Multi-stream tic pipelining | N/A (serial dependency) | - | - | - | - | rejected |
| 26 | Skip K2 when a batch has no spikes | trivial | automatic under 6 (8 us) | None | no | - | done |
| 27 | Skip constant per-launch traffic in baseline mode (rest, adapt, kc_mask loads; elig only in v6) | ~40% of the fixed 0.4 ms per launch while bandwidth-starved | elig made conditional on V6; rest/adapt still loaded | Low | no | - | M4 |
| 28 | Ring in a smaller dtype or ring row prefetch for the whole batch | halves ring traffic / hides latency | - | Medium | fp16 ring changes numerics; prefetch does not | conflicts with exactness if fp16 | M4 |
| 29 | Host power policy: raise GPU power limit / lock memory clock (`sudo nvidia-smi -pl`, `-lmc`), install `nvidia-powerd` | up to ~7x on every memory-bound kernel | blocked: needs root | Low | no | prerequisite for measuring 11-14 | open |

## Video path options

| Option | Expected | Measured | Risk | Interactions | Status |
|---|---|---|---|---|---|
| NumPy+Pillow compositor | 3-6 ms/frame at 720p | - | CPU contention with sim | baseline; required for tests | V1 |
| CuPy compositor | <1 ms GPU; frees CPU | - | shares GPU with sim: own stream + events | needs device `counts` | V3 |
| ffmpeg rgb24 pipe + h264_vaapi on the Intel iGPU | ~730 fps at 720p in the probe; zero load on the RTX | probe: 600 frames 720p in 0.82 s | render node must be the Intel one | primary; conflicts with PyNvVideoCodec zero-copy | V1 |
| ffmpeg rgb24 pipe + h264_qsv (oneVPL) | same iGPU with lookahead/ICQ | init fails until `libmfx-gen1.2` is installed | needs sudo | alternative to VAAPI | optional |
| ffmpeg rgb24 pipe + h264_nvenc | ~385 fps at 720p in the probe | probe: 600 frames in 1.56 s | shares the RTX | alternative; live sink | V1 fallback |
| Split encoders across GPUs (archive on iGPU, live on NVENC) | no single-engine queueing | - | two contexts to monitor | complements | V4 |
| PyNvVideoCodec zero-copy NV12 | removes D2H + swscale | - | context discipline; untestable in CI | needs CuPy compositor | V3 optional |
| H.264 vs HEVC/AV1 | HEVC halves archive size; AV1 encode absent on Ampere | - | browser HEVC support | H.264 live; HEVC archive optional | config |
| CBR vs VBR/CQ vs multipass | CBR for RTMP; CQ archive | - | NVENC load | multipass offline only | config |
| HLS fMP4 vs DASH vs LL-HLS | hls.js + Safari | - | LL-HLS weak in ffmpeg 6.1 | mediamtx relay for low latency | V4 |
| Local RTMP relay (mediamtx) | key out of argv; fan-out | - | one more service | complements tee | V4 |
| Separate CUDA stream (same context) | avoids serialising sim behind compositing | - | priorities/events | separate context rejected | V3 |
| Pinned frame buffers | 2-3x faster H2D/D2H | - | bounded pool | complements CuPy paths | V3 |
| 1080p canvas | legibility | - | 2x encode | config only | config |
