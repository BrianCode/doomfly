# Doom neural BCI — 2026-09-05

Public viewer: https://doomfly.example

Published version 3, source be03f5db93595298393e578d285d2dcfc4d5915d.
Anonymous HTTP checks returned 200 for the page and live feed. Two public
samples advanced from sequence 437 to 448, changed the actual input-frame hash,
and were under 500 ms old. Public WebMCP readback reported live BCI controls.
Verification is recorded in outputs/doom/public-verification.json. The local
development preview was stopped; the intended broadcaster and tunnel remain
running. At verification the simulation/wall-time ratio was 0.158.

Implemented: actual ViZDoom RGB → inferred R1–R6 receptor coordinates → full
retained MaleCNS v1.0 LIF graph → fixed readouts of DNp20/DNpe017 → game buttons.
The working BCI is an explicitly engineered mapping. It is not a claim about
natural fly motor commands, realistic vision, game understanding or learned skill.

The six-simulated-second closed-loop test produced 210 nonzero action tics and
three kills in one trial, taking 16.994 wall seconds on this host. Blacking out
pixels and disconnecting retinal outputs changed controls. Disconnecting every
edge abolished controls. Matched inputs were hashed. Seven focused neural/game
tests passed, including native-kernel comparison against the dense reference.
See outputs/doom/bci-validation.json and tests/test_doom.py.

The original biological-role comparison (DNa02/DNp09/MDN/MN9) remains visible;
it produced zero actions under the initial visual conditions. The BCI uses four
visually responsive descending cells, selected after visual-response calibration:
DNp20 R=10059, L=10162; DNpe017 L=10527, R=555871. The fixed gains are published.
No scene recognition, enemy positions, navigation policy or aim assistance is used.

The public spectator view includes actual input frames, retinal samples, neural
activity, readouts and applied controls, clocks, provenance, causal checks and
source. There is no recorded-frame fallback disguised as live. All spectators
watch the same run. UI tools affect only that viewer's overlays.

Positive game reward can deliver a labeled 200 ms artificial LB3c sugar stimulus.
It is disabled for the public baseline broadcast; counts and active state remain
visible. An exploratory one-pulse run later showed suppressed BCI readouts. That
uncontrolled comparison does not establish reward as the sole cause.
There is no synaptic-plasticity rule: stimulation is not training. No learning
result is claimed.

Remaining limitations: graded visual cells are approximated as LIF neurons;
display geometry, tonic lamina current and receptor calibration are assumed;
3,718 transmitter signs are uncertain. The model is slower than real time on
this host. The viewer reports actual speed, with neural/Doom clocks aligned
within 0.1 ms and no skipped neural steps. A fully realistic fly-brain player
and 1× real-time neural execution are not established by this implementation.

The broadcaster is a retained local process plus a temporary read-only HTTPS
tunnel. It becomes unavailable when the host sleeps/disconnects. 24/7 operation
requires an always-on compute host and stable tunnel/domain.
