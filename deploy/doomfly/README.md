# DOOMFLY multiday worker and viewer

The public stream now runs **experimental v6 training**, enabled at the user’s
request. The planned 72-hour observation window starts with this identified study;
it is not a forecast of successful learning. V6 failed its scientific gates.
The fixed baseline runs remain archived separately. See `docs/doom-live-training.md`
and the published `/live-training.json` for the exact protocol and failed checks.

## Current operating choices

- All 166,700 neurons and 25,582,938 directed edges, 0.1 ms neural integration,
  35 game tics per neural second and the existing fixed BCI are unchanged.
- Real 640×480 images, JPEG quality 75, up to eight captures per wall second.
  One-second immutable batches use ordinary CDN caching. A 2.5-second browser
  buffer plays original capture timestamps and matching telemetry. The viewer
  holds the last actual frame when no new state exists; it never invents motion.
- One shared worker, independent of spectators. One serialization per capture,
  request-scoped fetches and CDN caching of complete segments. In-flight promises
  are never shared between Worker requests. Browser polling stops while hidden
  and reconnects on return or network recovery; six-second client deadlines and
  four-second proxy deadlines cover both response headers and bodies.
  Detailed raster history accumulates locally from real received bins.
- Checkpoint every five wall minutes and on graceful shutdown; retain two
  generations. Full neural state, delayed events, weights and filtered controller
  rates are restored only against the identical source, graph, kernel and
  protocol. The baseline checkpoint format rejects learning subclasses; the new training
  checkpoint format delegates full neural, R8, memory and efficacy state to v6’s
  own checkpoint implementation, with atomic generations and integrity checks.
- Recovery starts a fresh Doom arena and new run ID linked to the saved run.
  The previous partial round is censored. ViZDoom 1.3.0 `save()` advances physics
  by two tics in the tested runtime, so it is not used for neural checkpoints.
  Recovery can lose work since the last checkpoint; outages are not training.
- All new step records are additionally stored in hourly gzip archives. The old
  100 MB rotating log alone was insufficient for a multiday scientific record.
  Ten-second flushes bound user-space buffering; abrupt failure can truncate the
  current gzip member. Keep interrupted periods labeled and back up the volume.

## Always-on compute

The Docker setup is prepared, **not cloud-deployed or container-tested here**:
this environment has no Docker daemon or connected compute account. Start with
4 vCPUs, 8 GiB RAM and 20 GiB persistent disk, then measure on that machine.
The neural kernel primarily uses one CPU thread; high per-core performance and
memory bandwidth matter more than simply adding cores. No GPU acceleration is
claimed. Keep BLAS/OMP at one thread to avoid contention.

From this project's root on a provisioned Linux host:

```sh
docker compose -f deploy/doomfly/compose.yaml up --build -d
```

The compose file mounts the audited graph and required annotations read-only;
these must already exist at the shown paths. The runtime verifies the graph
against `data-integrity.json`. Persistent `/state` stores recovery and audits.
The host port binds only loopback. Put a managed HTTPS proxy/tunnel in front of
the read-only service, configure monitoring and volume backup, and set the
viewer's server-side `DOOM_STREAM_ORIGIN` to that HTTPS origin. A healthcheck is
provided, but Docker's restart policy alone does not restart an unhealthy,
still-running process; an external supervisor must act on sustained staleness.
No credentials or graph files belong in the frontend bundle.

Before announcing continuous operation, verify on the actual host: cold start,
memory use, throughput, disk growth, public segment caching, load/fan-out, restart
recovery, interrupted-round labeling and the learning-validation gates.

## Vercel

The connected public site is a Sites/Vinext Cloudflare Worker, not a linked
Vercel project. No Vercel deployment was performed. Its wrapper routes call the
web-standard `doom-ui/lib/broadcast-proxy.ts`, so the same handler can be used in
Next.js on Vercel by supplying `process.env.DOOM_STREAM_ORIGIN` and omitting the
optional Cloudflare cache argument. It already emits Vercel CDN cache headers.
The existing Vinext build should not be uploaded as if it were a Next.js build.

Minimal Next.js route adaptations:

```ts
// app/api/broadcast/route.ts
import {broadcastProxy} from '@/lib/broadcast-proxy';
export const runtime = 'nodejs';
export const maxDuration = 10;
export function GET(request: Request) {
  return broadcastProxy(request, process.env.DOOM_STREAM_ORIGIN, '/index');
}
```

The dynamic `[run]/[segment]` route passes the validated
`/segments/${run}/${segment}` path to the same handler. Static pages/assets and
the client player are delivered by the website; continuous Python/C++ simulation
stays on the worker. A Vercel project URL and provisioned worker account/endpoint
are still needed to deploy there.

Vercel's current [function limits](https://vercel.com/docs/functions/limitations)
remain finite (800 seconds generally, 1,800 in the extended beta on eligible
plans). Workflow orchestration can span days, but that does not make a single
stateful neural/game process persistent. This setup uses brief web requests and
[CDN caching](https://vercel.com/docs/caching/cache-control-headers), not a
three-day Function invocation.

The Docker command now explicitly enables `--model experimental-v6 --learning`.
It uses `/state/training-v1` so old baseline checkpoints are never reused.
This container remains prepared but unbuilt and undeployed here. The running
public training worker is on the current Mac until a cloud target is supplied.
