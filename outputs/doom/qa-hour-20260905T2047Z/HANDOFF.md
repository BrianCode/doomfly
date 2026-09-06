# One-hour observation handoff

- Exact collection window: 2026-09-05 20:46:41.934 UTC through approximately
  21:46:41.934 UTC (3:46:41 PM–4:46:41 PM America/Chicago).
- Collector: `doom/observe.py`, exec session 41318, launched via `caffeinate -i`.
  Do not terminate it before `complete.json`. No neural model was restarted.
- Active simulation run: `67708368-3b24-439e-b45c-dd09d277eeef`, intact, BCI,
  reward off. The original broadcaster and tunnel remain running.
- Public companion probe: `doom/observe_public.py`, exec session 31684. It stops
  at the same fixed deadline. Its first valid sample occurred 103 seconds into
  the hour. The main Python HTTP client received 403; curl independently
  succeeded. Keep these distinct in availability analysis.
- Follow-up heartbeat: `fly-doom-one-hour-scientific-qa`, hourly, created around
  20:47 UTC. Finish analysis, deliver the report, then pause the heartbeat.
- Analyzer: `.venv-qa/bin/python -m doom.analyze_observation --out
  outputs/doom/qa-hour-20260905T2047Z`. The isolated environment is pinned by
  `analysis-environment.txt`. It does not alter the live neural environment.
- Preflight used a separate partial copy: 52 snapshots, 2,118 audit events, zero
  consistency violations. An intentionally changed turn command in another
  temporary copy was detected. `analyzer-preflight.json` records this check;
  it is not the one-hour outcome. Charts were exported and inspected there.
- Research and claim boundaries: `docs/doom-qa-prior-art.md` and the prior
  `docs/doom-neuroscience-review.md`. Existing matched control results must be
  retained in the verdict, even if this hour looks more favorable.
- `FIRE ON` means an attack button command, not a successful shot. Distinguish
  empty-ammo commands, actual ammo expenditure, kills and survival. We record
  no position/aiming ground truth, so cannot directly validate navigation.
- Write the final analysis to `docs/doom-one-hour-qa.md`; link the JSON, CSV,
  figures and supporting sources. Do not publish or announce externally.
- A passing technical traceability check does not make the approximate model
  biologically validated or a literal living fly brain. No plasticity exists.

- Performance investigation during observation: see `docs/doom-performance-review.md` and `outputs/doom/performance-20260905`. Brief probes/profiling and child-process retinal checks added host load; model/run unchanged. BLAS worker spin-waiting found; `doom/run_broadcast.sh` prepares a single-thread BLAS launch but has NOT been activated. Preserve the baseline before any restart or optimization trial.

- Separate learning candidate development since approximately 21:17 UTC adds substantial CPU load; see amendments.jsonl. No changes to the frozen baseline model/run.
