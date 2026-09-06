# Prior art and claim boundaries for the one-hour Doom QA

Research checked 5 September 2026. This is a targeted prior-art review, not an
exhaustive novelty search or independent experimental replication of other work.
The ongoing hour is defined in `outputs/doom/qa-hour-20260905T2047Z/protocol.json`.
No result from an incomplete observation should be presented as a final result.

## Relevant primary sources

| Work | What the primary source establishes | Implication for this project |
|---|---|---|
| [Shiu et al., Nature 2024](https://doi.org/10.1038/s41586-024-07763-9), [author code](https://github.com/philshiu/Drosophila_brain_model), [Oxford-hosted paper](https://ora.ox.ac.uk/objects/uuid%3A9422e222-dc23-4182-b183-19b92505a7db/files/rs1784n08p) | A connectome-based LIF model studied sensorimotor circuits, including feeding and grooming. The author implementation supports identified-cell stimulation and silencing. | Whole-brain LIF simulation and propagation from sensory to downstream cells are prior art. Its validated tasks do not establish fly vision or Doom learning in our different implementation. |
| [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3) | A connectome-constrained model of 64 visual cell types used task optimization to infer unknown parameters and compared predicted neural responses with experimental studies. Both the anatomical constraints and task optimization mattered for visual response predictions. | A wiring diagram alone is insufficient evidence that our uniform LIF visual pathway computes motion correctly. We need independent flash, contrast and motion-response validation. |
| [Eon technical explanation, 10 March 2026](https://eon.systems/updates/embodied-brain-emulation) | Eon describes a closed sensory–brain–body loop using existing connectome models and NeuroMechFly, acknowledges hand-chosen interfaces and body controllers, and states that internal biological signatures were not yet validated. | Connectome-based closed-loop embodiment is prior art. An entertaining embodied demonstration does not itself establish biological fidelity. |
| [FlyGM, February 2026 preprint](https://arxiv.org/abs/2602.17997) | The authors describe a whole-brain connectomic graph model for whole-body locomotion control. | Graph-constrained control is also prior art; this preprint is not evidence that our fixed LIF model has acquired a skill. |
| [Cortical Labs Doom demonstration](https://corticallabs.com/doom.html), [developer's public repository](https://github.com/SeanCole02/doom-neuron) | A prior Doom system interfaces with cultured biological neurons. Its repository describes trained stimulation encoding, configurable decoding and ablation experiments; these are the developer's claims, not independently reproduced here. | The broad concept of neurons controlling Doom is not new. Cultured living neurons and our numerical simulation are materially different substrates. We must not imply we have a living neural culture. |
| [Brunton et al., The digital sphinx, author-hosted manuscript](https://faculty.washington.edu/tuthill/docs/TheSphinx_2026.pdf) | The authors combine a worm connectome with a fly body and a trained motor interface to illustrate how realistic walking can occur despite an implausible biological mapping. They emphasize validation of the interfaces and biological predictions. | Behavioral resemblance is not sufficient validation of neural mechanisms. Our fixed decoder avoids a trained motor policy but its chosen mappings still require clear labeling and causal tests. |
| [Google Research, 3 September 2026](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/), [MaleCNS data](https://male-cns.janelia.org/download/) | The resource maps the adult male central nervous system. Our earlier independent data audit verified the official inputs and all retained runtime edges. | Credit the consortium for the connectome. Our contribution is an implementation using the released reconstruction, not creation of a brain or discovery of its wiring. |

## Defensible novelty position

The candidate contribution is an accessible, inspectable **MaleCNS-to-Doom
experiment**: retained graph coverage, a visible input-to-spike-to-button route,
a shared live feed, reproducible provenance, and openly reported limitations.
That is an engineering and science-communication contribution. Establishing new
neuroscience would require a novel, testable biological prediction and independent
experimental support. This search did not establish priority for this exact
combination; do not claim “first ever,” “first uploaded fly,” or a scientific
breakthrough based on the search returning no exact duplicate.

## Announcement claims to evaluate after the hour

**Potentially supportable after implementation checks:** “We connected a
simulation of the retained MaleCNS fly connectome to live Doom pixels and let
selected simulated neurons drive a fixed game interface. Watch and inspect the
experiment.” Explain approximate dynamics, artificial currents, inferred visual
geometry and engineered controls alongside that description.

**Unsupported now:** a literal living fly brain, an uploaded individual, natural
fly vision, understanding Doom, learned aiming, trained improvement, a biological
motor interface, or every living synapse/physiological state being reproduced.
Those are not wording preferences: the current implementation lacks the necessary
mechanisms and validation. The public model has no plasticity and its reward input
is disabled.

**Existing causal concern:** the earlier corrected-kernel, matched three-seed
six-second experiments recorded live-vision kills 1/1/1 versus black-input kills
3/3/2 and clamped-control kills 0/0/0. These are small descriptive trials, not a
benchmark. Nevertheless, they provide no evidence that live vision benefits
performance. Tonic lamina drive can sustain control activity without images.
The new hour must not erase or replace these results with a more favorable clip.

## Follow-up analysis instructions

Wait for the collector's `complete.json`, then run:

```sh
.venv-qa/bin/python -m doom.analyze_observation --out outputs/doom/qa-hour-20260905T2047Z
```

The isolated `.venv-qa` provides plotting without altering the active simulation's
environment. The primary observer's urllib public requests received HTTP 403
while independent curl requests returned valid live JSON. The companion probe
records those requests in `public-curl-checks.jsonl`; retain and distinguish both
transport outcomes. Neither a minute-sampled HTTP success nor a client-specific
rejection alone proves continuous public availability or an outage.

The final report should give separate verdicts for technical traceability,
biological validity, learning, gameplay value, novelty and announcement wording.
Do not change the live model during this frozen observation or call its partial
beginning/end episodes independent complete trials. The hour begins partway
through an already-running neural state and an already-running game episode.
