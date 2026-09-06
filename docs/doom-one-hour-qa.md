# DOOMFLY: completed one-hour QA

Observed 5 September 2026, 20:46:41.934–21:46:42.232 UTC
(3:46–4:46 p.m. Central). The original fixed-connection baseline was not reset
or modified during this window. This report concerns that baseline, not the
separate learning candidate developed alongside it.

**Verdict: the recorded pixel→neuron→button implementation is traceable, but the
run does not validate fly vision, learning, survival skill, or a literal fly
brain. It is suitable to describe as an inspectable connectome-based BCI
experiment. A learning/survival announcement is not supported.**

## Coverage and technical checks

- 3,600.31 seconds of observation; 716 state snapshots, 110 saved images and
  22,050 consecutive per-tic audit events. One run ID; no audit-tic gaps.
- Approximately 630 seconds (10.5 minutes) of neural/game time at 0.175× wall
  speed. This was not an hour of simulated fly experience. Maximum snapshot
  gap was 17.66 seconds, although the per-tic audit remained consecutive.
- No recorded consistency violations. Neural/game clocks matched within the
  declared numerical tolerance. Reconstructing the fixed decoder from rounded
  telemetry gave maximum errors of 0.000119 turn units and 0.000397 move units;
  rate-filter reconstruction differed by at most 0.000873 Hz.
- Source and graph hashes matched the running provenance and were unchanged
  across the window. These records do not contain a complete live weight or
  membrane-state checkpoint; they cannot prove every unobserved internal value.
- Activity stayed finite: 338,074,144 additional modeled spikes, averaging
  3.22 spikes/neuron/s across the retained graph. Population averages cannot
  establish that particular biological circuits behaved correctly.

## Gameplay and progress

There were nine complete game episodes and two boundary-censored episodes. All
nine complete episodes ended at the configured 60-second limit with health 100.
They produced two kills total (seven episodes with zero and two with one); a
third kill occurred in the final incomplete episode. This provides no evidence
of improved survival: every complete episode reached the same time limit, and
recorded health did not decline. It does not establish damage avoidance.

Ammo was empty in 84.4% of snapshots. An attack command was active while ammo
was empty in 76.4% of all snapshots. The button was pressed on 93.0% of audited
tics; that is not equivalent to firing a projectile or aiming successfully.
Forward output hit its clamp on 35.1% of tics. We did not record position or
aiming ground truth, so navigation cannot be validated from these traces.

Across thirds of neural time, mean forward output was 18.832, 18.848 and 18.807;
attack fractions were 93.05%, 92.75% and 93.18%. These are descriptive values
from one continuing neural state, not independent trials or evidence of
training. The public baseline has no plasticity and reinforcement is disabled.

The earlier matched six-second controls remain relevant: intact-vision kills
were 1/1/1 versus black-input 3/3/2 and clamped-output 0/0/0. They are too small
for a benchmark but show no demonstrated performance advantage from vision.
The separate new stimulus assays also found no modeled T4/T5 or KC spikes.
Those new candidate-development findings are documented separately in
[the learning review](doom-learning-review.md).

## Delivery and performance limits

The independent curl probe succeeded on 58 of 59 minute-spaced public checks.
One connection failed at 21:17:46 UTC. Successful public samples had median
frame age 0.294 seconds and maximum 2.479 seconds. All 59 requests by the primary
Python HTTP client were rejected with HTTP 403, a distinct transport result.
Neither series proves continuous availability or a continuous outage.

The host was shared with read-only profiling, separate full-graph experiments,
unit tests and website compilation. These loads are recorded in the observation
amendments; throughput is not an isolated hardware benchmark. The original feed
was restarted only after the completed observation was saved, to set BLAS to
one thread before NumPy import. That starts a new neural/game state and is not
part of this baseline. No neural equation, graph or timestep changed.

## Scientific validity and novelty

The retained MaleCNS reconstruction is real anatomical data. Its uniform LIF
approximation, inferred retinal geometry, tonic current and selected BCI outputs
are modeling/engineering choices. Graded visual signaling and many receptor or
modulatory mechanisms are not represented faithfully. The observed behavior
cannot validate those assumptions by resemblance alone.

[Shiu et al.](https://doi.org/10.1038/s41586-024-07763-9) previously modeled
sensorimotor functions with a fly connectome. [Lappalainen et al.](https://www.nature.com/articles/s41586-024-07939-3)
showed the importance of physiological constraints and parameter inference for
visual-response prediction. [Eon's embodiment demonstration](https://eon.systems/updates/embodied-brain-emulation)
and [Cortical Labs' neuron-controlled Doom](https://corticallabs.com/doom.html)
also precede this project. See [the primary-source prior-art review](doom-qa-prior-art.md)
for distinctions and additional work. This is not an exhaustive priority search.

Our defensible contribution is an accessible full-retained-MaleCNS-to-Doom
implementation with inspectable inputs, neural output traces, provenance and
openly reported negative controls. New neuroscience would require a new testable
biological prediction and independent experimental support. Do not claim first
ever, uploaded consciousness, a living brain, natural fly vision or learned Doom
skill from this demonstration.

Suggested wording: “We connected a simulation using the retained male fruit-fly
connectome to live Doom pixels. Selected simulated neurons operate a fixed game
interface. The experiment exposes its wiring, activity and validation limits.”

## Evidence

- [Machine-readable analysis](../outputs/doom/qa-hour-20260905T2047Z/analysis.json)
- [Observation figure](../outputs/doom/qa-hour-20260905T2047Z/observation.png)
  and [PDF](../outputs/doom/qa-hour-20260905T2047Z/observation.pdf)
- [Timeseries CSV](../outputs/doom/qa-hour-20260905T2047Z/timeseries.csv)
  and [episode CSV](../outputs/doom/qa-hour-20260905T2047Z/episodes.csv)
- [Protocol](../outputs/doom/qa-hour-20260905T2047Z/protocol.json),
  [completion record](../outputs/doom/qa-hour-20260905T2047Z/complete.json),
  [amendments](../outputs/doom/qa-hour-20260905T2047Z/amendments.jsonl)
- [Comprehensive implementation audit](doom-neuroscience-review.md)

The figure and exported data were generated from the completed recording. The
figure was visually inspected. The analyzer previously detected an intentionally
corrupted action record in a separate negative-control fixture.
