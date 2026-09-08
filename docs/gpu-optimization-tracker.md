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
| 2026-09-08 | v6 server, `--backend cutile --learning --resume` from the converted CPU checkpoint, before the lazy mirror | speed 0.29x, `brain_step_ms` 89 (six pageable state downloads per <=10 ms bin, 3-4 bins per tic) |
| 2026-09-08 | v6 server, same, with the lazy host mirror (only spike counts downloaded per bin) | **speed 1.000x, `brain_step_ms` 16.3-16.8** over 60 s, still power-capped; CPU v6 kernel on this host would be about 45 ms/tic |
| 2026-09-08 | v6 server on the GPU with `--video-mp4` (1280x720 archive, VAAPI on the Intel iGPU), first compositor (FreeType text, fancy-index resize) | compositing 25-29 ms/frame on the video thread; the interpreter lock it held cut the sim to **0.876x**, `brain_step_ms` 22.5 |
| 2026-09-08 | same with the glyph-atlas compositor (all pixel work in NumPy) | compositing 3.4-5.6 ms/frame, encoder 35.4 f/s, 0 drops, sim back at **1.000x**, `brain_step_ms` 16.4-16.8 |

## Host power state (resolved 2026-09-08 with a memory clock lock)

Measured 2026-09-08 under a sustained CuPy copy load: pstate P5, SM 1762 MHz,
**memory clock 810 MHz of 5501 MHz max**, power 22 W against a current limit of
25 W (default 35 W, max 65 W), throttle reasons `SW Power Cap` and
`SW Thermal Slowdown` active at 51 C, PCIe link gen 1 x8. Achieved bandwidth:
6-14 GB/s for plain CuPy copies (device peak is about 190 GB/s); a raw CUDA
kernel doing the same work as the cuTile evolve kernel was equally slow, so the
kernels are not the cause.

Resolution: `sudo nvidia-smi -lmc 5001,5501` locks the memory clock (verified: P0,
5501 MHz, no throttle reasons, 115 GB/s CuPy copies, 28 W). `sudo nvidia-smi -pl 35`
is refused on this laptop ("not supported"), and `nvidia-powerd` is not packaged
for it. The lock does not survive a reboot; re-apply it (or a udev/systemd unit)
before GPU runs. Still open: the PCIe link stays at gen 1 x8 (2.5 GT/s of a 16 GT/s
maximum) even during transfers, so host<->device copies run at 1.6 GB/s; a
0.67 MB array costs ~0.4 ms each way (ASPM policy is `default`; changing it needs
root and was not tried). The kernel figures below are split into before and after
the clock lock.

## Kernel time budget

Before the clock lock (milestone 1, memory clock 810 MHz):

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

After the clock lock (memory clock 5501 MHz), tuned defaults TN=1024, TE=128:

| Component | Measured | Note |
|---|---|---|
| K1 `evolve_batch`, 1 substep / 18 substeps | 0.079 / 0.30-0.33 ms | per-substep ring cost ~11-14 us (1.3 MB at ~115 GB/s) plus ~0.06 ms fixed |
| K2 `deliver`, real v6 batch (1,054 spikes, 122 k edges) | 0.20 ms | no hub tail, no target contention (max 46 arrivals to one target) |
| K2, baseline bench batch (1,547 spikes, 0.24 M edges) | 0.076-0.084 ms | |
| Baseline `step()` per 286-substep tic (17 k spikes/tic) | 7.7 ms -> 6.9 ms with the drive assembled on device | `doom.compare_backends`: GPU 7.8 ms median vs CPU 30.7 ms, speedup 2.67x |
| Host<->device per call | drive H2D removed; counts D2H 0.41 ms (PCIe gen 1) | |
| v6 server (`--learning --video-mp4`) | `brain_step_ms` 17.7-18.3, speed 1.000x | 3-5 bins per tic; profile per 100-substep bin: 6 batches 4.9 ms GPU, plastic upload 0.10, device drive 0.17, counts D2H 0.41 ms |
| v6 `rgb_step` wall incl. retina sampling and rate rule | 15.3 ms neural, 19.7 ms total per tic (bench, 28.6 k spikes/tic) | before the device rule |
| v6 `rgb_step` with the rate rule on the device (one download per tic) | 13.9 ms neural, 16.2 ms total per tic | nsys: deliver 6.1 ms + evolve 5.4 ms per tic, GPU ~80% busy inside the brain step: now GPU-bound |
| v6 server unpaced (`--speed 0`) | 1.32x realtime with the video archive, 1.38x without; `brain_step_ms` 15.5 | the rest of the tic is ViZDoom, two SHA-256 hashes (~2-3 ms), retina sampling, decode |
| int32 fixed-point ring (`DOOM_CUTILE_RING=int`) | 6.89 ms/tic median, same as float32; 3 repeated runs bit-identical; oracle passes | see `docs/gpu-numerics-exploration.md` |

Tile sweep after the lock (K1 18 substeps): TN 256: 0.360, 512: 0.333, 1024: 0.302, 2048: 0.330 ms.
K2 (bench batch): TE 128/grid 4096: 0.076; TE 256/grid 2048: 0.084; TE 512/grid 1024: 0.103; grid 8192: 0.100 ms.

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
| 8 | Lazy host mirror (D2H v,g,ref,adaptation,eligibility only on access/checkpoint) | ~1 ms/tic | **89 -> 16.5 ms/tic** on the power-capped host (pageable 0.5 ms per array x 6 arrays x 3-4 bins) | Low | No | `server.py:161` reads `brain.v` once per publish | M3 done |
| 9 | Pinned host buffers + async copies | 0.1-0.2 ms/tic; pageable copies measured 0.4-0.55 ms each | implemented (drive, counts, mirrored state) | Low | No | complements 22 | M1 done |
| 10 | Counts stay on device across v6 bins | ~0.2 ms/tic | - | Low | No | - | M4 |
| 11 | Tile/occupancy tuning (TN, TE, grid, hints) | 10-30% kernel time | after the clock lock: TN=1024 -9% on K1, TE=128/grid 4096 -10% on K2; hints untested | Low | No | - | M4 done |
| 12 | CUDA graph / stream capture per distinct `steps` | 0.3-0.5 ms/tic | - | Medium | No | requires 6 | M4 |
| 13 | Edge-balanced delivery (row split across `DELIVER_SPLIT` CTAs, implemented as a 2-D grid) | cuts the 11k-degree tail | real v6 batches have max degree ~2.5 k and no tail; the split adds empty CTAs (see the K2 sweep) so it defaults low; an earlier 2.2 ms K2 reading was a profiling artifact | Medium | No | complements 14 | M4 done, kept configurable |
| 14 | CSR locality: sort `post` within rows; neuron permutation for atomic locality | 10-30% of K2 | - | Medium | rounding order only | complements 13 | M4 |
| 15 | Integer fixed-point ring (Q10 contact units, int32 atomics; plastic weights quantised to 1.3e-4 mV) | deterministic run-to-run at no cost | implemented as `DOOM_CUTILE_RING=int`: same speed as float32, oracle passes, repeats bit-identical; float32 stays the default for CPU-like sums | Low | yes (last bits of g; documented) | alternative to 16; conflicts 18 | done, optional |
| 16 | Deterministic sort + segmented sum | determinism at 0.5-1 ms/batch | - | Low | yes | superseded by 15 | not recommended |
| 17 | Weight code uint16 + LUT | halves weight reads | - | Medium | no | conflicts plastic scatter | not now |
| 18 | fp16 weights | bandwidth only | - | Low | yes | - | not recommended |
| 19 | Sleeping-neuron / sleeping-tile skip | ~none (ring traffic dominates) | - | Medium | no | - | not recommended |
| 20 | Batch > 18 via refractory | impossible (delay-limited) | - | - | - | - | rejected |
| 21 | Pull/CSC delivery | no benefit vs atomics | - | High | rounding | - | rejected |
| 22 | Drive assembled on the device from the same recipe (lamina, retina, sugar, tonic, pulses in the same float32 order); retina sampling and the low-pass stay on the CPU | removes the 0.67 MB drive upload per call (0.4 ms at PCIe gen 1) | baseline 7.7 -> 6.9 ms/tic; v6 unchanged (transfer was not its bottleneck) | Low | No (identical values; `light` hash unchanged) | complements 9 | M4 done |
| 23 | Rate rule on the device (CuPy float64, same closed form) | the FLOPs are nothing, but it removes a download, a host update and an upload per <=10 ms bin | v6 neural time 15.3 -> 13.9 ms/tic; total 19.7 -> 16.2 | Low | ulp-level float64 differences in `memory_w` (tests use rtol 1e-9) | supersedes 31 | done |
| 24 | Move 10 Hz JPEG/JSON publish off the sim thread | 5-10 ms per publish | - | Low | no | independent of GPU; the video service already moves compositing off-thread | later |
| 25 | Multi-stream tic pipelining | N/A (serial dependency) | - | - | - | - | rejected |
| 26 | Skip K2 when a batch has no spikes | trivial | automatic under 6 (8 us) | None | no | - | done |
| 27 | Skip constant per-launch traffic in baseline mode (rest, adapt, kc_mask loads; elig only in v6) | ~40% of the fixed 0.4 ms per launch while bandwidth-starved | elig made conditional on V6; rest/adapt still loaded | Low | no | - | M4 |
| 28 | Ring in a smaller dtype or ring row prefetch for the whole batch | halves ring traffic / hides latency | - | Medium | fp16 ring changes numerics; prefetch does not | conflicts with exactness if fp16 | M4 |
| 29 | Host power policy: lock memory clock (`sudo nvidia-smi -lmc 5001,5501`) | up to ~7x on every memory-bound kernel | K1 2.15 -> 0.33 ms, K2 0.39 -> 0.08 ms, baseline tic 43 -> 7.7 ms; `-pl` unsupported on this laptop | Low | no | prerequisite for 11-14 | done (re-apply after reboot) |
| 30 | PCIe link stuck at gen 1 x8 (1.6 GB/s) | 4-8x on the remaining host<->device copies (~0.4 ms per call) | measured under sustained transfer | needs root (ASPM policy / BIOS) | no | complements 10 | open |
| 31 | One spike-count download per game tic (counts accumulate on the device across bins) | ~0.4 ms x (bins-1) per tic | done together with 23 | Medium | no | needs 10 | done |
| 32 | Wall-clock pacing control (`--speed`, 0 = unpaced) | exposes the headroom: sim no longer sleeps ~10 ms per tic | 1.32-1.38x realtime for v6 | None | no | - | done |
| 33 | Move the two per-tic SHA-256 digests (0.9 MB frame, 0.67 MB counts) and the audit line off the sim thread | ~2-3 ms of the ~6 ms per tic spent outside the brain step when unpaced | - | Low (hashlib releases the GIL) | no | independent | later |
| 34 | Neuron permutation for delivery locality (targets of one presynaptic cell contiguous) | delivery is 6 ms/tic of random atomics into a 12 MB working set (L2 is 2 MB) | - | Medium | rounding order only | same as 14 | later |

## Parallel optimization round (2026-09-08, one worktree per item, combined on `opt-combined`)

| Item | Outcome | Measured | Notes |
|---|---|---|---|
| 1 Substep-major delivery | rejected | K2 0.170 -> 0.194-0.252 ms; v6 tic 13.9 -> 18.1 ms | delivery is bound by atomic-unit throughput (~0.7 G atomics/s), not ring locality; per-substep scans in the evolve kernel are expensive at full clock. Only fewer atomics can help (combine same-target arrivals) |
| 2 Off-thread audit hashing | see below | | |
| 3 CUDA graph capture | works, not adopted | v6 13.8 -> 13.8 ms, baseline 6.9 -> 6.8 ms | cuTile launches capture and replay correctly; the device-side time origin is kept in reserve for when GPU time shrinks enough for launch gaps to show |
| 4 Neuron permutation (reverse Cuthill-McKee, cached) | **adopted** | K2 0.170 -> 0.140 ms; v6 13.9 -> 12.2 ms/tic; baseline 6.9 -> 6.7 ms | bandwidth median 31,882 -> 17,206; `DOOM_CUTILE_ORDER=identity` restores graph order; ~100 MB extra host memory |
| 5 Block-wise ring loads | rejected | K1 16 substeps 0.251 -> 0.228 ms (~0.27 ms/tic) | the per-substep ring cost is exactly bandwidth (11.6 us for 1.3 MB); only fewer bytes would help, and fp16/int16 are rejected |
| 6a Skip v6-only state in the baseline evolve kernel | **adopted** | K1 0.302 -> 0.248 ms; baseline tic 6.9 -> 6.0 ms | v6 unchanged |
| 6b/6c Fused drive scatter, tonic dirty flag | rejected | ~0.25 ms/tic, below noise | the fused scatter also needed a pinned-buffer pool with events to avoid a race |
| 7 Receptor sampling | **adopted as a CPU lookup table** | sampler 0.47 -> 0.17 ms per call (two calls per tic); v6 total 16.2 -> 15.7 ms | bit-identical on all 111 recorded frames; the device version lost because the 921 KB frame upload over the gen-1 PCIe link costs 0.56 ms |
| 8 Platform | **adopted** | | `deploy/doomfly/nvidia-memory-clock.service`, `gpu-clocks.sh`, PCIe notes |
| Combined (4 + 6a + 7 + 8) | | baseline 5.9 ms/tic; v6 12.1 ms neural, 13.6 ms total; K2 0.140 ms; unpaced server **1.61x** realtime, `brain_step_ms` 13.4 | 108 tests pass in both ring modes |

## Video path options

| Option | Expected | Measured | Risk | Interactions | Status |
|---|---|---|---|---|---|
| NumPy compositor with a pre-rendered glyph atlas | 3-6 ms/frame at 720p | 3.4 ms (bench), 5.6 ms (live); FreeType text per frame was 11 ms and the fancy-index resize 4.5 ms | holds the GIL: a 26 ms compositor cost the sim 12% | baseline; required for tests | V1 done |
| Compositor + encoder in a separate process (pipe or shared memory) | removes all GIL contention with the sim thread | - | 1.6 MB/tic over a pipe (55 MB/s) | complements everything above | later |
| CuPy compositor | <1 ms GPU; frees CPU | - | shares GPU with sim: own stream + events | needs device `counts` | V3 |
| ffmpeg rgb24 pipe + h264_vaapi on the Intel iGPU | ~730 fps at 720p in the probe; zero load on the RTX | live: 35.4 f/s sustained, 1.5 ms write per frame, no sim impact; note: `-count_frames` reports 2 fewer frames than written at EOF (B-frame flush), to investigate | render node must be the Intel one | primary; conflicts with PyNvVideoCodec zero-copy | V1 done |
| ffmpeg rgb24 pipe + h264_qsv (oneVPL) | same iGPU | works after `libmfx-gen1.2` with `-low_power 1` only (lookahead rejected); 600 frames 720p in 0.81 s; live server: 35.2 f/s, 0 drops | needs sudo for the package | alternative to VAAPI; auto order is vaapi, qsv, nvenc, x264 | done |
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
