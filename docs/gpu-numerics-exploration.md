# Integer and low-precision arithmetic for the GPU kernel

Question: could the simulation run in integer math, or in fp8 or a similar reduced
format, and would that make it faster? Measured 2026-09-08 on the RTX 3050 Laptop
(sm_86, memory clock locked at 5501 MHz) with the v6 model state after ten game
tics of the calibrated brain, and on the oracle in `tests/test_doom_reference.py`
(exact spike bins, membrane and synaptic state within 0.002 mV).

## What the numbers must represent

| Quantity | Observed range | Precision the oracle needs | Notes |
|---|---|---|---|
| Membrane `v` | -323 to -45 mV; 2% of neurons within 1 mV of threshold at any instant | 0.002 mV at -45 mV | threshold crossings decide spike bins |
| Synaptic drive `g` | -500 to +530 mV; p99.9 of \|g\| 114 mV | 0.002 mV | exponential decay every substep |
| Adaptation | 0 to 15 mV | same | KC only |
| Fixed weights | contacts x 0.275 mV, 1 to 2,591 contacts, 99.997% of edges | exact | int16 contact counts represent them exactly |
| Plastic weights (4,184) | baseline x [0.1, 2.0], not multiples of a contact | ~1e-4 mV | need a fractional unit |
| Ring cell (sum of arrivals to one neuron in one substep) | up to 532 mV = 1,936 contacts; 293 k non-zero cells per 18 substeps | 0.002 mV | this is what the atomics accumulate |

## Formats

| Format | Verdict | Why |
|---|---|---|
| fp8 (e4m3, e5m2) | not available | cuTile gates float8 to sm_90+; the RTX 3050 is sm_86. Even on Hopper, 3 to 4 mantissa bits cannot hold a 0.002 mV membrane tolerance |
| fp16 / bf16 state (`v`, `g`) | rejected | fp16 spacing at 45 mV is 0.031 mV (15x the tolerance); bf16 is 0.25 mV |
| fp16 ring | rejected | spacing at 532 mV is 0.5 mV; 92% of observed ring cells are not exactly representable; and the evolve kernel got slower (0.33 vs 0.25 ms per batch) because of the conversions, while delivery gained only 8% |
| int16 ring, Q4 (1/16 contact) | rejected | the largest cell seen (1,936 contacts) is already within 6% of the int16 limit (2,047 contacts); saturation would silently change spikes |
| int16 ring, Q8 | rejected | 0.25% of cells overflow today |
| **int32 ring, Q10 (1/1024 contact, 2.7e-4 mV per unit)** | **adopted as an option** | exact for every fixed weight, quantises plastic weights to within 1.3e-4 mV, holds +-2.1 million contacts per cell (1,100x the largest observed sum), converts to float32 exactly below 2^24 units (4,500 mV). Integer atomics are associative, so sums are bit-identical regardless of arrival order |
| int32 fixed-point state (`v`, `g`) | possible, no benefit | same 4 bytes as float32; the exponential decay needs a 64-bit intermediate multiply; the only gain would be cross-platform bit reproducibility, which the int ring already gives for the part that was order-dependent |
| int16 contact-count weights | no speed benefit | halves weight bytes, but delivery is bound by atomic throughput and latency, not by weight reads (measured: same time with int16 and float32 weights) |

## Measured kernel times (full graph, synthetic batch of 1,100 spikes)

| Variant | evolve, 18 substeps | deliver | ring size |
|---|---|---|---|
| A float32 ring, float32 weights (default) | 0.251 ms | 0.242 ms | 20 MB |
| B int32 Q8 ring, int16 weights | 0.252 ms | 0.236 ms | 20 MB |
| C fp16 ring, float32 weights | 0.332 ms | 0.223 ms | 10 MB |

The kernels are limited by the ring's memory traffic (evolve) and by random
atomics (deliver), not by arithmetic. Halving bytes with fp16 does not pay for
the conversions, and integers cost nothing extra.

## What integer math buys, and what it does not

- Determinism: with `DOOM_CUTILE_RING=int` the conductance sums no longer depend
  on the order in which atomics land, so two runs from the same checkpoint produce
  identical spike counts, and `spike_counts_sha256` in the audit log becomes
  reproducible on the GPU. (Two float32 runs happened to agree over 100 tics in
  `doom.compare_backends`, but that is luck, not a guarantee.)
- Accuracy: exact integer sums are at least as accurate as the CPU kernel's
  sequential float32 additions; the oracle passes in both modes.
- Not speed: the same bytes move and the same atomics execute. The remaining
  speed levers are structural (fewer host launches, ring locality, delivery
  locality through a neuron permutation), listed in `docs/gpu-optimization-tracker.md`.
- Compatibility: the int ring changes the last bits of `g` compared with the
  float32 mode, so a run should keep one mode for its lifetime; the mode is
  recorded in the kernel build record and therefore in checkpoint identity.

## How to use

```sh
DOOM_CUTILE_RING=int bash doom/run_gpu.sh --backend cutile ...      # deterministic mode
DOOM_CUTILE_RING=float32 ...                                        # default, CPU-like float32 sums
```
