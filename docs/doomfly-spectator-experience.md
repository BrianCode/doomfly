# DOOMFLY spectator experience

Product direction requested 5 September 2026: scientifically defensible live
training that is immediately legible and entertaining, with minimal interaction
and useful reasons to share or return. This is a launch design proposal, not a
claim that persistent validated training is already deployed. The current public
broadcast is the fixed baseline; the learning candidates remain unvalidated.

## Core experience

Everyone joins the same ongoing experiment. Opening or closing a browser does
not create, restart or accelerate a brain. The featured run has a stable identity,
model version and accumulated simulated training time. Game deaths start new
episodes; the specified learned state persists according to the documented model.
Fast neural-state resets, memory decay and checkpoint restores must be disclosed.

The first viewport, on mobile and desktop, should provide:

1. Actual Doom gameplay, visible immediately without sign-in or a start button.
   Keep the gun and hazard cues unobscured. Silent playback by default, with
   optional sound where the browser supports it.
2. A compact phase label: TRAINING, EVALUATION, REPLAY, BASELINE or PAUSED. Only
   show LIVE while the feed is fresh. An evaluation with frozen weights is not
   ongoing training, even when it is being broadcast live.
3. Health, current round survival and episode count. One plain-language cue
   explains the task: blue floor damages the avatar; gray floor is safe.
4. One honest progress result: the latest fixed-protocol comparison with frozen
   controls, its uncertainty and when it was measured. Allowed outcomes include
   improved, inconclusive and worse. While evidence is pending, say so.

The compact black-and-white pixel-art theme and scientific/gamer voice remain.
Detailed neurons, memory changes, sensory fields and methods are optional views;
understanding the primary task must not depend on opening them. A suitable
continuity message, once the model implements it, is: "Same brain. Another round."

## A reason to stay, return and share

- Immediate story: a visible hazard, an escape opportunity and consequences.
  Pick and validate an arena that produces interpretable decisions; do not add
  corrective actions, a hidden navigation policy or unrecorded resets for drama.
- A short event ribbon can report measured events such as a health decrease,
  the engineered dopamine stimulus, an actual efficacy update or a round end.
  A weight update is never labeled proof of learning. Avoid claims that the
  simulated brain felt pain, understood Doom or became conscious.
- Longer story: accumulated training experience, deaths, completed benchmark
  checkpoints and generalization results. Show both real elapsed time and
  simulated brain time, clearly labeled. Keep outages separate from active
  training time. Do not use a fictitious "percent learned" meter.
- A short latest-moment replay provides context when someone arrives during a
  quiet stretch. Keep it secondary to the live view and label its capture time,
  run/model identity and playback speed. It must not silently masquerade as live.
- Share links or short clips should preserve a real recorded event and include
  enough context to understand it. Highlights illustrate behavior; the complete
  results establish whether learning occurred. Retain failures and regressions.
- A scheduled benchmark gives returning viewers a meaningful milestone. The
  comparison schedule and decision criteria are fixed before looking at results.
  There is no guarantee that any particular checkpoint will show improvement.

Initial interaction budget: watch immediately; optional unmute, replay/share and
"How it works." Audience actions must not alter the scientific training run.
Field and activity overlays are spectator-only. A later participatory experiment
would require its own declared protocol and separate data.

## Duration and biological time

The completed v6 pilot processed 80.2282 neural seconds in 490.740 seconds of
recorded episode wall time, approximately 0.1635×. On comparable hardware/load:

| Simulated experience | Approximate wall time |
| --- | --- |
| 1 hour | 6.1 hours |
| 24 hours | 6.1 days |

These are throughput extrapolations, not predictions of when learning succeeds.
They exclude startup and extra control/evaluation replicas. The timing includes
episode warmups; it is not a pure interactive-experience rate. Faster hardware,
network activity and contention can change it materially. Twenty wall seconds
currently contain only about 3.3 neural seconds at this measured average.

The number of trials needed for valid survival learning remains unknown. The
current failed candidates do not justify an hours/days/weeks success forecast.
Extending a run cannot repair failed sensory or conditioning validation.

Biological learning and retention times depend on the circuit and protocol.
[Aso and Rubin (2016)](https://doi.org/10.7554/eLife.16135) compare brief and longer
conditioning, repeated training, and retention across different dopamine-defined
compartments. A multiday webcast does not establish multiday biological memory.
Our model's decay and any future consolidation must have separate evidence.
Do not slow the scientific clock or alter learning constants to prolong a show.

## Evidence behind the spectator story

Before describing the product as a fly model learning to survive:

- Validate useful sensory responses, post-cue recovery and cue-specific
  conditioning with appropriate timing and no-imposed-reinforcement controls.
- Establish survival improvement with independent training replicas and unseen
  tests against frozen and timing-shuffled arms. Preserve all predefined results;
  do not pick the most successful run after seeing outcomes and call it typical.
- Evaluate frozen snapshots of the featured training state so the benchmark
  does not itself teach to its test. Keep a genuinely held-out final test set;
  repeated public benchmarks are monitoring data and can become development data.
- Test retention and whether memory erasure or the intended sensory intervention
  removes benefit, accounting for general impairment. Report uncertainty using
  independent training replicas, not thousands of correlated frames as samples.
- Preselect the featured run or a declared rotation of replicas. The one
  entertaining broadcast is not a substitute for the independent research cohort.
- Freeze the model, fixed decoder and task protocol within a reported experiment.
  A physiological fix or a new task begins a separately labeled version with
  fresh baselines. Do not splice incompatible histories into a learning curve.
- Update and verify the published "How it works" against that exact release,
  including the learning mechanism, training/evaluation phases and limits.

## Multiday operation and launch checks

The current broadcaster depends on the local host staying awake and connected.
Before promising continuous multiday training, move computation to suitable
always-on infrastructure, with durable checkpoints, run/episode logs and a
verified restart path. A restart must either reproduce the saved brain and game
state or explicitly start a new episode with documented retained memory.

Keep the simulation independent of spectator count. Encode/fan out a shared
stream and timestamped telemetry; do not run a full brain per visitor. A stream
can display a real frame repeatedly while waiting for the next one, but cannot
claim 30/60 new game states per second if the simulator is slower. Clearly label
time-compressed replays. Never synthesize neural activity or game outcomes to
make the feed look faster. Outages display a stale/paused state rather than a
replay labeled live.

Set product targets before launch: first genuine frame quickly on representative
mobile connections, no required interaction, immediate task comprehension, and
readable stats without obscuring gameplay. Test these with new viewers. Measure
time to first frame, short-session retention, returning viewers and sharing;
virality is a product hypothesis, not a promised outcome.

Muted playback with an optional unmute control is consistent with
[Chrome's autoplay policy](https://developer.chrome.com/blog/autoplay/).
Mobile playback and recovery still need checks in the actual supported browsers.
No browser testing or infrastructure migration was performed as part of this
design note.
