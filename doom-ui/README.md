# Current viewer status

The shared broadcast is experimental v6 training. It remains scientifically
unvalidated. See [the exact protocol](../docs/doom-live-training.md) and
[setup instructions](../README.md). The notes below document the original viewer
and earlier baseline; the current source is [on GitHub](https://github.com/nftechie/doomfly).

# DOOMFLY viewer

A public, read-only view of one shared MaleCNS v1.0 / ViZDoom connection
experiment. The fixed visual-neuron BCI turns, moves, and fires. The biological-role
walking/mouthpart comparison remains silent. No learned game skill or validated
fly vision is claimed. Every page and broadcast keeps these limits explicit.

The server route `/api/live` proxies only the configured broadcaster `/state`.
`DOOM_STREAM_ORIGIN` is set through Sites; `.dev.vars` is local-only and ignored.
No browser inputs, credentials, or variable URLs are forwarded. The edge caches
frames for one second to reduce fan-out load. A stale broadcast becomes offline;
there is no prerecorded fallback. The UI shows the actual simulation/wall ratio.

[The GitHub repository](https://github.com/nftechie/doomfly) contains the simulation,
learning experiments, viewer, tests and scientific reviews. `methods.json` and
`validation.json` expose provenance and historical results. The bundled source
archives remain historical snapshots; GitHub is the current source entry point.

Checks completed: TypeScript, production build, seven neural/game boundary tests,
four full-graph conditions, and both WebMCP tools with valid/invalid inputs and
state readback. Broad browser visual testing was not requested.

The live broadcaster runs on the user's machine through a temporary public
read-only tunnel. It is unavailable if the host sleeps/disconnects; 24/7 service
requires an always-on compute host and stable tunnel.

## Pixel UI redesign

Monochrome arcade layout, Silkscreen typography, original 16×16 SVG interface
glyphs, and one generated pixel-art fly in `public/pixel-fly.png`. The fly is a
static decorative illustration, not a model-state visualization. Generated with
imagegen on 2026-09-05 for this site; no source footage was used.

The live Doom image, polling, decoder, broadcast route, and neural/game runtime
are unchanged. Retinal and neural charts are now monochrome. Detailed readouts,
model limits, reinforcement notes, causal checks, and raw audit data are retained
in collapsed disclosures. The two viewer-only overlay tools remain available.
