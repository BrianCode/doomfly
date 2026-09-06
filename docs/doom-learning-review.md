# Learning candidate: implementation and first validation

5 September 2026. This is a development record, not a claim of successful fly learning.

The experiment now has a separate full-graph memory implementation, controlled
stimulus and conditioning assays, a real ViZDoom survival task, held-out tests,
frozen/shuffled reinforcement arms, retention/reset controls, brain checkpoints,
per-tic provenance, and a website comparison page. The original live baseline
and its hour-long observation remain separate.

## Scientific verdict

**The candidate does not yet learn to survive. Do not announce it as a learning
fly brain.** Two prerequisite failures were measured before interpreting game
scores:

- Eight controlled visual stimuli each ran for one neural second. Photoreceptors
  and the engineered BCI respond, but all 6,865 T4 cells, 6,720 T5 cells and 4,064
  Kenyon cells produced zero spikes. All 206 annotated KCg-d cells were silent.
  Opposite images therefore did not produce distinguishable memory-cell spike
  patterns. This concerns our spiking approximation; it does not establish that
  graded biological signals are absent in real flies.
- Directly imposing activity on disjoint KC ensembles bypasses vision. It causes
  depression with the candidate rule, but loses cue selectivity. Changes also
  occur with backward pairing and without imposed dopamine stimulation, because
  endogenous modeled dopamine neurons fire. The numerical rule passes its unit
  checks while the full-network conditioning assay fails. This is not a
  reproduction of the original odor-conditioning experiment.

A follow-up current sensitivity check at 7.2, 8 and 10.5 mV equivalent still
produced widespread KC activity and endogenous PPL101 spikes without an imposed
US. Lowering the input amplitude alone did not establish selective conditioning.
These chosen currents are not calibrated physiological measurements.

## Model and anatomical scope

All 166,700 retained MaleCNS v1.0 neurons and 25,582,938 directed edges remain.
Plasticity is restricted to 4,184 existing KC→MBON11 edges. MBON11 IDs are 10704
and 11402; PPL101 IDs are 11327 and 11900. The reconstruction identifies cell
anatomy, not receptor localization or synaptic learning parameters.

The candidate adds a presynaptic eligibility trace and dopamine-gated depression,
with no postsynaptic spike requirement. This is inspired by
[Hige et al.](https://doi.org/10.1016/j.neuron.2015.11.003), not a literal parameter
copy or experimental replication. [Aso and Rubin](https://doi.org/10.7554/eLife.16135)
show why a single depression rule cannot represent all fly memory compartments.
The exact equations, chosen constants, identities and inferred compartment
assignment are documented in [the implementation protocol](../doom_learning/README.md).

Two PPL101 cells now deliver a modeled modulatory channel, replacing their
baseline default fast excitatory effect. All outgoing edges are retained, but
receptor responses outside the chosen memory rule remain unknown. Every control
arm uses the same candidate dynamics, so a comparison with the public baseline
alone would confound that change with plasticity.

## Game and experiment

ViZDoom's existing `health_gathering` scenario provides a damaging floor and
health pickups. It is a real Doom-engine survival arena, not a new task invention.
An engineered adapter converts a decrease in health into next-interval PPL101
stimulation. Pixels are the sensory input. Health, object identities and game
rewards never enter the fixed neuron-to-button decoder. Evaluation switches off
reinforcement and plasticity, retains trained efficacies, and resets fast state.

The initial 12-second pilot uses two training seeds and two distinct evaluation
seeds, with learning-on, frozen, shuffled, retention and memory-reset conditions.
It is a pipeline diagnostic with a short censoring horizon, not a powered study.
One reconstructed animal repeated across game seeds does not provide independent
biological animals. Final episode results are in
`outputs/doom-learning/survival-pilot/results.json` when complete.

The initial shuffled schedule circularly shifts a zero-padded, horizon-length
sequence. Early death can truncate its delivered dose, which the analysis flags
as a failed control. That failed pilot must be preserved. Future scheduling must
shift the observed exposure interval and still check actual delivered dose; even
then earlier death can break matching. A changed survival mean alone is never a
learning result.

## Development speed

The headless runner eliminates browser delivery, JPEG compression and real-time
sleep. It still generates every sensory image and integrates every 0.1 ms neural
step. Independent replicas can run in separate processes or on different hosts.
Checkpoints save complete brain state, not the external game engine. A scheduler
benchmark compares complete input/spike/action traces across serial and parallel
execution. No faster-than-real-time claim is made without measurements.

Do not accelerate the learning rate, coarsen the integrator, skip game frames,
remove weak edges or replace the network with a policy and call that the same
scientific experiment. Those changes would require separate validation.

## Next prerequisite

A calibrated visual and cell-dynamics model must reproduce contrast/motion
responses and sparse, distinguishable visual KC activity before another training
campaign is meaningful. [Connectome-constrained visual-system modeling](https://www.nature.com/articles/s41586-024-07939-3)
used physiological constraints and optimized unknown parameters; anatomy alone
was insufficient. Its smaller, different visual graph cannot be substituted for
MaleCNS and presented as our full retained brain. Any transfer of cell physiology
requires explicit assumptions and independent response tests.

After that gate passes, conditioning must show cue specificity and timing
controls in the full network. Only then should longer, independently seeded
survival experiments test generalization, retention and loss of benefit after
memory reset. The present results justify that next research step; they do not
justify claiming that it has already been accomplished.

## Completed development results

The initial pilot completed 22 episodes, totaling 260.91 neural seconds in
1,240.76 seconds of episode wall time (0.210× on the shared host). All six training
runs recorded zero KC spikes and zero changed memory edges. All 12 held-out
episodes hit the 12-second cap, so their eventual survival time is unknown.
Retention and reset controls also had no learned efficacy change to preserve or
erase. This is a negative mechanism result, not evidence of equal lifetime
survival between treatments.

The observed-interval scheduling correction was subsequently run in the real
game on both original training seeds. Delivered US matched paired exposure
exactly: 78/78 and 82/82 game tics. Neither corrected trial changed any memory
connections. These follow-up records repair the scheduling check, not the
failed vision or conditioning model, and do not replace the original pilot.
See `outputs/doom-learning/exposure-correction/results.json`. Future runs now
refuse to overwrite an existing experiment, capture exact sources before their
first episode, record initial health/game time, and keep separate image folders
for each training episode.

A serial/parallel/parallel/serial scheduling check ran two independent seeds per
batch for one neural second each. All eight recorded input/spike/action/health/
memory traces matched their corresponding seed exactly. Parallel batch time
totaled 24.86 seconds versus 27.84 serial (about 1.12× aggregate throughput in
this short shared-host measurement, including setup). Individual episode speed
ranged from 0.117× to 0.506×. The earlier one-direction comparison appeared faster,
but startup and host load were confounders; it is not a reliable 2.8× claim.
The headless/scheduling machinery is implemented, but 30/60 real neural-driven
FPS and faster-than-real-time full-brain simulation have not been achieved.

Technical verification: 42 passing relevant tests across the baseline, graph
import and memory/control suites. The website builds and type-checks. New lab
components pass targeted lint; the whole repository still has existing lint
violations in other components. No numerical check overrides the failed
scientific gates.

Checkpoint restore now also rejects mismatched original weights, retinal geometry, input-cell assignments and compartment gains/masks. Earlier pilot checkpoints must be used with their saved source snapshot; the stricter current format intentionally rejects missing configuration identity.
