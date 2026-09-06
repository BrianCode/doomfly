# Learning candidate iteration log — 5 September 2026

The announcement claim remains unearned. This log preserves unsuccessful
candidates as well as useful corrections; none has replaced the public fixed
brain broadcast. Every candidate uses the complete retained MaleCNS v1.0 graph
(166,700 neurons; 25,582,938 directed edges). A simulated reconstruction is not
a living fly brain.

## Candidate v2: transmitter channel and visual input corrections

`doom_learning_v2` separates all 541 cells annotated dopamine, octopamine or
serotonin from the generic fast excitatory channel. All their reconstructed
outgoing edges deliver a contact-weighted modulation trace. Only the selected
PPL101 / KC→MBON11 rule has an implemented postsynaptic plasticity effect.
Receptor-specific effects at other targets and possible cotransmission remain
unknown. This is an explicit model assumption, not proof those cells have no
fast effects in vivo. No neuronal entries or edges are removed.

The new visual adapter includes all 330 annotated R8p and 481 R8y cells.
Coordinates are inferred from the modal column among **all** contacts to
column-annotated targets, using the original R1–R6 display transform. Median
contact agreement is 1.0; 22 cells have agreement below 0.8. Coordinates are
not measured retinal viewing angles. Linear sRGB blue/green are engineering
proxies for R8p/R8y light drive; they are not calibrated spectral photon flux.
Unknown and ultraviolet-sensitive subtypes remain in the graph without invented
optical inputs. Photoreceptors still use a spiking proxy for graded biology.

390 existing R8→aMe12 edges receive a positive net sign, retaining their original
contact-weight magnitudes. R8 acetylcholine/histamine cotransmission and excitation
of AMA cells (including aMe12) are supported by [Xiao et al., 2023](https://doi.org/10.1038/s41586-023-06681-6).
The aMe12→KCγ-d visual memory route is described in
[the visual-input study (2024)](https://doi.org/10.1038/s41467-024-49616-z) and
[the photoreceptor-target study](https://doi.org/10.7554/eLife.71858).
The type correspondence is transferred to the annotated male cells; the exact
synaptic strength and receptor distribution of this male are not measured.

At one second, a blue RGB image produced 283 spikes in 14 of 206 KCγ-d cells;
black produced none. Left and right blue half-fields recruited 10 and 5 cells.
These are signal-propagation results, not proof of useful visual behavior.
Green failed to activate this particular memory population. Its absence is a
remaining model limitation, not evidence that living flies cannot see green.

Direct KC conditioning at 7.2 mV-equivalent current still failed the original
selectivity/timing criterion: paired MBON output increased, and no-imposed-US
and backward conditions also changed a small number of synapses. Endogenous
DAN activity and recurrent feedback require examination. Importantly, biological
DANs can respond to sensory cues without an imposed punishment, so a zero-change
control expectation is itself incomplete for some paradigms. We will not simply
relax the criterion after observing a failure: a replacement assay must state its
predictions and controls before the next outcome is collected.

Raw records: `outputs/doom-learning/physiology-v2/`; frozen sources and hashes are
in `source-snapshot/` and `source-manifest.json`. Nine kernel/checkpoint tests
passed; numerical correctness does not establish biological correctness.

## Candidate v3: KC resting-potential sensitivity

`doom_learning_v3` tests a KC-specific resting/reset potential of −60 mV instead
of applying the baseline −52 mV to every neuron. Other baseline thresholds,
time constants and contact magnitudes are retained. A −60 mV wild-type KC example
is reported in [Greenin-Whitehead et al., 2025](https://doi.org/10.1113/JP288790).
This is a coarse physiological sensitivity test, not a fit to all KC subtypes,
not a reproduction of that paper's compartmental model, and not a parameter
selected for Doom scores. Its consequences for visual recruitment and conditioning
must both be measured. Direct stimulation uses 15.2 mV-equivalent, 0.2 mV above
the new rheobase, to match the v2 relative current protocol.

The direct-stimulation assay now leaves backward and no-imposed-US weights
unchanged. Its initially chosen learning rate still gives no detectable output
suppression. An explicitly exploratory learning-rate sweep of 0.005, 0.02 and
0.05 produced paired MBON spike counts of 13, 13 and 1, from 13 before training;
the unpaired count stayed at 90. These are calibration data, not held-out
validation. The stronger suppression is qualitatively compatible with the
[Hige et al. conditioning study](https://doi.org/10.1016/j.neuron.2015.11.003),
but direct current injection is not odor presentation or calibrated optogenetics.
Some visual patterns still recruit broad KC populations while others fail to
reach the selected output cells. This candidate is not ready for a behavior claim.

## Candidate v4: intrinsic spike adaptation

`doom_learning_v4` adds a decaying, spike-triggered inhibitory current only to
KCs. Its exact subthreshold solution is integrated alongside the original LIF
currents at 0.1 ms; no spikes, cells or reconstructed contacts are skipped.
The adaptation principle is consistent with sensory sparseness modeling
([Farkhooi et al., 2013](https://doi.org/10.1371/journal.pcbi.1003251)). The 8 mV
increment and 200 ms decay are **chosen sensitivity-test values**, not measured
MaleCNS KC parameters. That paper's generic insect model does not validate this
adult male implementation. The cell-specific resting-potential assumption is
retained from v3. Eleven integration, causality and checkpoint tests pass.

Adaptation reduced white-stimulus KC spikes from 36,169 to 7,860, but still
recruited 1,475/4,064 KCs, predominantly gamma-main and alpha/beta types; only
three KCγ-d cells fired. APL remains a spiking proxy despite its known local,
graded physiology. Sparse visual-memory recruitment and broad indirect KC
recruitment should not be conflated.

The completed eight-branch assay counterbalanced which blue stripe orientation
was paired with reinforcement. Both forward-paired and no-imposed-US training
suppressed both cue responses to zero. Frozen responses were identical before
and after training. The predeclared cue-selectivity gate failed.
The separate v2 left/right cue assay also failed its counterbalanced gate.

## Candidate v5: calibrated background activity and centered plasticity

[Huang, Luo et al. (2024)](https://doi.org/10.1038/s41586-024-07819-w)
provides measured spontaneous activity and a baseline-centered, bidirectional
memory model. The downloaded Figure 1 source data give PPL1 gamma1pedc
20.09 ± 4.30 Hz and MBON gamma1pedc 37.17 ± 9.06 Hz (mean ± sample SD,
20 cells each). Figure 2 shock responses give 50.14 ± 15.05 Hz (12 cells).
Workbooks, extraction ranges and hashes are in `research/huang-2024`.

Candidate v5 fits constant background currents to these resting rates before
any Doom scoring: MBON current 9.87 and PPL current 11.3125 mV-equivalent give
approximately 37/37 and 23/19 Hz in the two modeled cells. These are fitted
background proxies, not measured intrinsic pacemaker mechanisms. A 200 ms
additional PPL current of 4 gives 40/40 Hz in a dark calibration; 6 gives
55/60 Hz. The conditioning assay predeclared and used 4 throughout.

The rate rule adapts the paper's Appendix equations 3.2–3.5 onto all 4,184
existing KC→MBON11 edges, using actual full-network spikes. It does not replace
the brain with the paper's nine-unit model. KC and dopamine traces drive
bidirectional efficacy changes; dopamine rates are centered on the measured
baseline. Efficacies retain positive signs. One-second traces, 50 ms weight
filter, 30-minute passive memory decay and 0.1–2.0 bounds are stated model
choices, not individually validated parameters of this male reconstruction.
The neural integrator remains 0.1 ms, with rates updated at up to 10 ms bins.

Numerical checks verify pairing-order effects, wrong-compartment controls,
passive decay, frozen weights, and exact checkpoint continuation. A fixed-rate
10/5/1 ms comparison constrains rule integration error; it does not establish
convergence of the entire coupled neural/visual system.

**The complete counterbalanced conditioning assay failed.** With the left cue
paired, its MBON response fell by 93.79%, but **the no-imposed-US control showed
exactly the same output suppression**. The opposite cue fell by 4.88% in both.
With the right cue paired, suppression was greater for the wrong cue. Erasing
weights restored the original responses and frozen weights remained unchanged,
but these technical successes do not rescue the conditioning result.

A more fundamental problem appeared in the traces: after left-cue presentation,
the frozen model produced 276,709 KC spikes in the final dark second, with PPL
rates around 313 Hz. This persistent recurrent state cannot be treated as a
validated representation or a physiological reinforcement response.

## Candidate v6: combine adaptation with centered memory

Candidate v6 combines v4 KC rest/adaptation assumptions with v5 centered memory
and background currents. It preserves all neuronal IDs, directed edges and
contact magnitudes except the explicitly documented R8 target-specific sign
and learned KC→MBON efficacies. Four additional checkpoint/reset/native-parity
checks pass. A frozen-network stability protocol was saved before collection:
2 seconds dark, 1 second cue, 3 seconds recovery.

The model has no KC spikes during a dark baseline. After left blue, right blue
and white cues, final recovery KC rates are respectively 8,950, 8,935 and 8,976
spikes/s across the population. Corresponding PPL rates are 106/105, 108/95 and
108/105 Hz, versus approximately 22/18 Hz before the cue. Adaptation reduces
excessive activity but does not validate post-cue recovery or sparse visual
coding. This candidate therefore remains exploratory and is not promoted to
the live broadcast.

## Actual Doom survival arena and headless experiment

`doom_learning/survival_arena.py` generates a two-sector UDMF Doom map with an
original blue hazard flat and a gray safe flat. ViZDoom evaluates damage
(20 health every 32 game tics on the hazard); no Python survival policy exists.
Seeded spawn positions and headings are independent of neural actions. A
programmed **environment-only** test confirms that remaining on blue causes
death at 128 tics (3.657 s), while crossing to gray survives the 12 s test.
Those programmed controls are not used by the fly and are not presented as its
behavior. The engine has already applied an initial 20-health hit when the first
controllable frame is available; observed episodes start at 80 health.

`doom_learning_v6/survival.py` runs the entire pixel→full brain→fixed BCI→game
loop without the browser, video compression or wall-clock sleeps. Game health
only schedules the declared aversive input into PPL101; coordinates are recorded
for the observer and never select controls. A pulse lasts exactly 2,000 neural
steps (200 ms) and can end inside a game tic. Timing-shuffled exposure is shifted
at neural-step resolution and actual delivered durations are checked.

The exploratory pilot predeclares one training seed (41041), two held-out starts
(61041 and 61042), one training episode and an 8 s horizon. It compares learning
on, frozen, and shuffled timing; evaluation freezes weights and removes imposed
reinforcement. The trained branch also tests black vision, a five-second rest,
and memory erasure. It is a technical pilot after failed scientific gates,
**not** an independent validation campaign. More trials cannot turn its upstream
physiological failures into evidence of a literal fly learning to survive.

### Completed survival pilot

All 12 episodes completed. Learning-on changed 1,862 connections;
shuffled timing changed 1,896; frozen changed zero. All three training arms
delivered exactly 600 ms of PPL stimulation before death. On held-out seed
61041, learning-on and shuffled both died at 3.657 s, while frozen reached
the 8 s cap with 60 health. On seed 61042 all arms died at 3.657 s.

Erasing memory restored the **entire** frozen input/spike/action/health/position
trace on seed 61041, including survival to the cap. Thus memory changes caused
a behavioral difference in this model, but the observed difference was harmful.
Black vision with the learned weights lasted 7.314 s on that start; this does
not establish useful vision. A five-second rest still ended at 3.657 s.
No general causal effect size, significance, or fly population inference follows
from one training replica and two held-out starts.

All nine mechanical integrity checks pass: exact observation counts, clock
alignment within half a 0.1 ms step, frozen efficacy identity, no imposed US in
tests, exact exposure accounting, matched shuffled dose, input/spike hashes,
complete episode records and memory-erasure trace identity. The combined
relevant Python suite reports **68 passing tests**, including baseline reference
integration, graph import, rule dynamics and checkpoint causality.

Recorded episodes processed 80.2282 brain seconds including warmup in
490.74 wall seconds (0.163×). Neural integration used
91.9% of episode time. These shared-host timers exclude model
construction and the separate five-second retention interval. The experiment
is already headless; rendering every game frame remains necessary. Larger
learning rates, skipped frames, coarser timesteps or pruned edges would change
the experiment and cannot be advertised as equivalent acceleration.

The announcement gate remains **NOT READY**. The immediate scientific blocker
is calibrated cell and visual dynamics with stable cue representations and
proper recovery, followed by independent conditioning. The fixed BCI also needs
validation that neural valence changes support the chosen motor task; its
arbitrary game mapping cannot be mistaken for natural fly behavior. Longer
training or more favorable selected seeds would not repair these failures.

## Remaining evidence for announcement

Before any survival claim, predefine a visual-conditioning assay with active,
distinct cue representations, compatible reinforcement timing, frozen and
shuffled controls, retention and memory reset. Then test a fixed neural decoder
in an actual Doom hazard task using independent training replicas, unseen test
situations, exposure accounting and confidence intervals. The benefit must depend
on the learned weights and intended sensory pathway; ablation and erasure tests
need controls for general impairment. Improvements in training scores or internal
weight changes alone do not qualify.

Before a launch green light, publish an updated "How it works" page explaining
the exact deployed sensory, neural, reinforcement, learning and action loop.
Distinguish training from evaluation or the fixed baseline, document what changes
and what stays fixed, and link the supporting evidence and limitations. Verify
the published explanation against that release's code and configuration. Keep
the explanation brief and accessible, with the site's scientific/gamer voice.
