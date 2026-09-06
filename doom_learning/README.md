# DOOMFLY learning laboratory

This laboratory is separate from the public baseline. A working numerical
learning rule is not a validated fly model or proof of improved survival.

## Fast development

Use the existing Python 3.11 environment, with BLAS configured before import:

```sh
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning.vision
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning.conditioning
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning.run \
  --out outputs/doom-learning/survival \
  --seeds 41031,41032,41033 --eval-seeds 61031,61032,61033 \
  --train-episodes 2 --episode-seconds 30
```

The survival runner has no wall-clock sleep, browser, JPEG encoding, large
broadcast history, or per-frame global weight hashing. Every 640×480 game
observation still passes through the original retinal transform. Each game
tic is preceded by 285/286 neural steps of 0.1 ms, keeping the clocks aligned.
Rendering the sensory image is necessary and is not skipped. `--frames` saves
occasional genuine input images for review. It does not alter the neural input.

Recorded timing separates neural-kernel, sensory-transform, and game costs.
No claim of faster-than-real-time execution is made before measurement.
Independent seed runs can be distributed to separate machines without changing
their protocols. More concurrent runs on this laptop can slow every run; use
serial experiments here. Changing the timestep, sensory cadence, graph, or
learning rate is a new scientific condition, not a transparent speed upgrade.

Brain checkpoints include all mutable neuron, queue, eligibility, modulation,
and weight arrays and verify model/graph identity on restore. They resume the
brain, not the external Doom engine. Held-out episodes deliberately start a
new game and clear fast neural state while retaining trained efficacies.

## Candidate mechanism

The complete retained MaleCNS v1.0 graph contains 166,700 cells and 25,582,938
directed edges. The candidate uses all 4,184 existing KC→MBON11 edges. The
annotated target cells are MBON11 10704 and 11402; the reinforcement cells are
PPL101 11327 and 11900. Raw IDs, indices, instances, and mask hashes are exported.
No target-seeking policy, new graph edge, circuit crop, or learned decoder is used.

The candidate is a phenomenological, dopamine-gated depression rule inspired
by gamma1/peduncle memory physiology. At each KC spike, a presynaptic eligibility
trace increases by one and decays exponentially with a chosen 1,000 ms time
constant. When a PPL101 spike arrives after the original 1.8 ms delay:

```
eligibility_i(t) = sum over prior KC spikes exp(-(t - spike_time)/1000 ms)
w_ij <- max(0.1 * original_w_ij,
            w_ij * exp(-eta * anatomical_DAN_fraction_j * eligibility_i(t)))
```

The update occurs at the existing 0.1 ms neural timestep, requires no MBON
spike, and changes only the eligible existing connections. The default eta is
0.001. Eta, trace duration, weight floor, and current amplitudes are explicit
chosen parameters, not measured values or a fit to Doom success. Freezing
plasticity leaves the circuit dynamics and stimulus path in place.

Two PPL101 neurons deliver a modulatory channel instead of the baseline's
unsupported default fast excitatory sign. All their outgoing edges contribute
to a target-specific transmitter trace. The effect on KC→MBON11 efficacy is
modeled; receptor effects at other targets remain unresolved. Each MBON's
relative dopamine participation is inferred from its two direct PPL101 contact
weights. Aggregate neuron-to-neuron data do not locate dopamine receptors at
individual contacts. This inference requires validation and is not a measured
release model. Other unresolved physiological features of the baseline remain.

Hige et al. demonstrated compartment-specific, presynaptic, dopamine-dependent
depression with timing dependence. Aso and Rubin showed that memory rules differ
across compartments and can include rewriting and forgetting. Our LTD-only
candidate does not model that whole repertoire, long-term consolidation, an
animal's motivation, or a reward-prediction-error circuit.

Sources:
- https://doi.org/10.1016/j.neuron.2015.11.003
- https://doi.org/10.7554/eLife.16135
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4135349/
- https://male-cns.janelia.org/download/

## Validation gates

1. **Vision.** Controlled brightness, static patterns, and opposite motion pass
   through the exact baseline retinal transform and full graph. Compare cell
   responses, especially T4/T5 and KCs, before interpreting a game score. Raw
   image sensitivity in an engineered BCI does not establish useful fly vision.
2. **Mechanism.** Independent numerical tests check event timing, compartment
   specificity, no-dopamine and frozen controls, checkpoint equality, and reset.
   The full-network imposed-KC assay tests qualitative cue selectivity and
   forward/backward pairing. It bypasses sensory encoding and cannot count as
   a reproduction of an odor experiment or successful visual conditioning.
3. **Survival.** ViZDoom's packaged `health_gathering` arena uses damaging floors
   and health kits. Damage schedules an engineered 200 ms PPL101 current pulse
   on the next game interval. This is an experimental unconditioned stimulus,
   not a validated nociceptor pathway or a claim that a simulated animal feels
   pain. Health and game rewards never enter the fixed neuron-to-button decoder.
4. **Learning.** Match training seeds across plastic, frozen, and shuffled-US
   arms. Evaluate with reinforcement/plasticity off on separate seeds. Record
   actual stimulation exposure, censoring, quiet-delay retention, and memory
   reset. Survival changes alone, checkpoint persistence, or a changed weight
   hash are insufficient to pass. Record failures as failures.

The simple survival scenario is reused from ViZDoom, not newly invented:
https://vizdoom.farama.org/environments/default/

The first controlled stimuli produce no spikes in the current model's T4/T5
and KC populations, including the 206 annotated KCg-d cells. Thus this candidate
cannot currently form visual memory through that path. The full imposed-activity
assay also tests whether excessive recurrent activity destroys cue specificity.
Successful toy-network checks do not override full-network failures.

Repairing this limitation requires a calibrated sensory/cell-dynamics model.
Published visual-system modeling found that detailed wiring plus constrained
physiological/task parameter optimization mattered for motion tuning; uniform
cell dynamics are insufficient. Importing a smaller trained visual network as
an undisclosed substitute would violate this project's graph requirement.
https://www.nature.com/articles/s41586-024-07939-3

## Parallel execution and reproducibility

```sh
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning.sweep \
  --out outputs/doom-learning/parallel-run --jobs 2 \
  --seeds 41031,41032 --eval-seeds 61031,61032
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning.benchmark \
  --out outputs/doom-learning/scheduling-check --seconds 1
```

Use a fresh output directory per experiment. The first pilot is frozen in
`outputs/doom-learning/survival-pilot`, including its exact source snapshot.
Its raw conditioning label `no_dopamine` means **no imposed US**, not dopamine
neurons silenced. The current source and viewer use the accurate label.

The public experiment record reports failed gates rather than a learning claim.
See `docs/doom-learning-review.md` for measured failures and the next physiological
calibration requirements. Numerical tests, checkpoint persistence and parallel
execution equivalence establish technical properties only.

Checkpoint compatibility is strict. The current format also verifies original weights, retinal geometry and compartment masks/gains. Use the exact saved pilot source snapshot for older checkpoints lacking those signatures.
