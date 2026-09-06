# DOOMFLY performance investigation — 2026-09-05

The running baseline was not restarted or reconfigured. These measurements
diagnose the present system; they are not a demonstration of 30 or 60 FPS.

## Observed throughput

On the current Apple M1 Pro (10 CPU cores, 16 GiB RAM), a roughly 25-minute
segment of the ongoing QA recording advanced 7.28 game tics per wall second
(0.208× simulation speed) and published 5.17 frames per second. A recent
293-second segment advanced 6.84 tics/sec and published 4.99 frames/sec.

The latest 60 five-second telemetry samples reported a median native-kernel
duration of 75.1 ms per 28.6 ms simulated game interval, mean 121.2 ms and
maximum 1158.6 ms. These are periodic, time-sampled observations of the most
recent step, not an exhaustive per-tic timing profile or unbiased CPU fraction.
The host was also used for development during this observation.

A separate 16-request public probe received 15 distinct states over 8.81
seconds between first and last completion: 1.59 state changes per second.
Median curl request duration was 277 ms (range 201–489 ms). Median frame age
at reception was 258 ms, maximum 1751 ms. Curl was launched separately per
request, with a 150 ms pause; this does not measure browser paint FPS or model
the browser's connection reuse exactly. Evidence: `outputs/doom/performance-20260905/public-delivery.json`.

## Bottlenecks

1. The native neural kernel is serial and advances 285/286 substeps for each
   game tic. Its graph, delays and 0.1 ms timestep are not pruned or skipped.
2. The image-to-luminance transform calls OpenBLAS for 3-channel products.
   A one-second, 10 ms interval native stack sample found nine BLAS helper
   threads largely spin-waiting and the simulation thread also waiting inside
   `cblas_sgemv64_`. A separate process-usage snapshot showed about 427% CPU for
   the broadcaster and 1% for the Doom process. The stack sample includes 52/80
   simulation-thread samples in `neural_advance` and 14/80 in matrix multiply.
   These are short diagnostic observations, not a full-run attribution.
3. `doom/server.py` publishes no more often than every 100 ms: a 10 FPS ceiling
   even with a faster brain. The observed publication rate is nearer 5 FPS.
4. The client waits for each full JSON/image response and then another 150 ms.
   Each public response in the probe was about 167–169 kB, including full
   raster history and repeated metadata, not just a video frame.
5. The edge handler requests a one-second cache TTL where the Cache API is
   available. This is a configured source of potential repeated frames, not
   an established 1 FPS cap: most probe responses had distinct sequences.

## Prepared improvement; not activated in the baseline

`bash doom/run_broadcast.sh` starts the same server with
`OPENBLAS_NUM_THREADS=1` set before NumPy imports. All command-line arguments
are forwarded. It does not stop any existing process. Starting it on an
occupied port is not a replacement/restart procedure.

A separate-process 1-thread / 10-thread / 1-thread check ran the unchanged
`retinal_samples` function on eight recorded JPEG frames with the actual
retinal UV mapping. Mean times were 0.493 / 38.065 / 0.892 ms per conversion.
All eight output arrays had identical SHA-256 hashes across configurations.
This isolates a large overhead in that transform; it does not imply the
whole brain or broadcast will speed up by the same factor. These were JPEG
test inputs, not a replay of the exact original raw RGB sensory stream.
Evidence: `outputs/doom/performance-20260905/retina-threading.json`.

After preserving the baseline, benchmark the launcher in a separate run,
record thread configuration, compare neural traces, and measure stage-level
timings before adopting it as the public runtime. The sampled retinal equality
is encouraging but does not substitute for the end-to-end comparison.

## Path to 30 FPS

- Remove BLAS oversubscription and profile the remaining neural hot paths.
  About a fourfold increase over the measured 7.28 tics/sec is needed to
  produce 30 distinct game states per wall second; 35 tics/sec is normal speed.
- Replace repeated image-plus-history polling with a persistent broadcast
  channel. Publish every produced input frame, synchronize frame identifiers
  with telemetry, and transmit slower charts separately. Keep a shared fanout
  service so spectator count does not multiply requests into the brain host.
- Benchmark native-kernel improvements or dedicated CPU/GPU execution against
  the numerical reference. A GPU port is a candidate, not a measured speedup;
  sparse synaptic updates and floating-point accumulation need validation.
- Report produced-frame FPS, received-frame FPS and simulation speed separately.

ViZDoom's normal game logic runs at 35 tics/sec. Our synchronous adapter
advances one tic per neural interval and currently paces at most normal speed.
A 60 FPS display would require separate render interpolation or duplicated
frames, a renderer supporting that path, or a changed simulation protocol;
60 distinct neural decisions/sec is not a presentation-only setting. Rendering
between recorded states must be distinguished from frames supplied to neurons.

Primary documentation: https://vizdoom.farama.org/api/cpp/doom_game/

No live neural source, weights, decoder, sensory transform, timestep, or
reinforcement mode changed during this investigation. Read-only HTTP probes,
the one-second stack sampler and short child-process retinal benchmarks add
some host load; the hour is an observational QA run, not an isolated benchmark.

## Post-observation application, 5 September 2026

After the frozen hour completed at 21:46:42 UTC, the verified broadcaster was
restarted with `doom/run_broadcast.sh` and `OPENBLAS_NUM_THREADS=1`. The old run
and its observations remain preserved. The new run ID is
`9f09fe4c-8f1b-4842-93c7-0e799940b27b`; its state starts fresh. No graph, neural
equation, timestep, sensory mapping or decoder changed.

A later 20.23-second local-source sample advanced 13.40 neural seconds:
0.662× speed and 8.10 published source frames per wall second. This is a short
sample on a shared host, not a controlled before/after experiment or browser
paint FPS. It does not establish 30/60 FPS. Raw samples and the restart record
are in `outputs/doom/performance-20260905/post-restart-sample.json` and
`post-observation-restart.json`.
