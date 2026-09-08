"""cuTile (NVIDIA GPU) implementation of the fixed-step LIF model.

Same graph, constants and per-substep schedule as doom/kernel.cpp and the
dense reference in doom/engine.py: integrate and threshold, deliver the spikes
emitted 1.8 ms earlier (dropped for refractory targets), then reset. Every
neuron is evolved densely on every 0.1 ms substep, which the dense reference
already proves equivalent to the lazy CPU kernel.

Batching: a spike emitted at substep t first changes another neuron's
conductance at the end of substep t+18, so up to 18 substeps are evolved per
kernel launch with no inter-neuron communication. Spikes of one launch are then
scattered into a per-substep conductance ring (float32 atomics) for the next
launch. The 2.2 ms refractory period bounds spikes to one per neuron per launch.

Numerics: float32 like the CPU kernel. Arrival sums use atomics, so the
accumulation order is unspecified; the spike counts and states match the
Brian2 oracle to its declared tolerance but are not bit-identical to the CPU
kernel and can differ slightly between runs (see docs/gpu-optimization-tracker.md).
"""
import hashlib
import math
import time
from pathlib import Path
import numpy as np
from doom.engine import Brain

ROOT = Path(__file__).resolve().parents[1]
try:
    import cupy as cp
    import cuda.tile as ct
    AVAILABLE = True
    IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - exercised only without a GPU stack
    cp = ct = None
    AVAILABLE = False
    IMPORT_ERROR = error

import os
TN = int(os.environ.get('DOOM_CUTILE_TN', 1024))    # neurons per evolve tile (power of two)
TE = int(os.environ.get('DOOM_CUTILE_TE', 128))    # edges per delivery iteration (power of two)
RING = 32          # conductance ring slots, power of two >= 2*DELAY
DELAY = 18         # 1.8 ms / 0.1 ms
REFRACTORY = 22    # 2.2 ms / 0.1 ms
THRESHOLD = -45.0
DELIVER_GRID = int(os.environ.get('DOOM_CUTILE_GRID', 2048))
DELIVER_SPLIT = int(os.environ.get('DOOM_CUTILE_SPLIT', 4))   # CTAs sharing one spike's row (tail of 11k-edge hubs)
MODEL_REVISION = 'lif-r2-refractory-write-protection-cutile'
# Conductance ring numerics: 'float32' reproduces the CPU kernel's float32 sums (atomics make the
# order unspecified); 'int' accumulates fixed-point contact units so every sum is exact and the
# result is independent of the order of arrivals (bit-reproducible run to run).
RING_MODE = os.environ.get('DOOM_CUTILE_RING', 'float32')
RING_INT = 1 if RING_MODE == 'int' else 0
CONTACT_MV = 0.275                           # mV per synaptic contact in this graph
RING_Q = 10                                   # fixed-point fraction bits below one contact
RING_UNIT = CONTACT_MV / (1 << RING_Q)        # 2.7e-4 mV per unit; int32 holds +-2.1e6 contacts per cell


def _log_capacity(npad):
    # At most one spike per neuron per 18 substeps, so 2*npad entries always
    # retain every spike of the last 18 substeps (needed for checkpoint export).
    return 1 << int(math.ceil(math.log2(2 * npad)))


if AVAILABLE:
    @ct.kernel
    def evolve_batch(v, g, ref, adapt, rest, drive, kc_mask, elig, elig_last, ginc, counts,
                     log_neuron, log_step, log_counter,
                     t0: ct.int32, nsteps: ct.int32, a: ct.float32, b: ct.float32, coup: ct.float32,
                     c: ct.float32, k_adapt: ct.float32, jump: ct.float32, elig_scale: ct.float32,
                     LOG: ct.Constant[int], V6: ct.Constant[int], RINGI: ct.Constant[int], unit: ct.float32):
        blk = ct.bid(0)
        ids = blk * TN + ct.arange(TN, dtype=ct.int32)
        idx = (blk,)
        vt = ct.load(v, index=idx, shape=(TN,))
        gt = ct.load(g, index=idx, shape=(TN,))
        rt = ct.load(ref, index=idx, shape=(TN,))
        at = ct.load(adapt, index=idx, shape=(TN,))
        rest_t = ct.load(rest, index=idx, shape=(TN,))
        dt = ct.load(drive, index=idx, shape=(TN,))
        cnt = ct.zeros((TN,), dtype=ct.int32)
        spike_at = ct.full((TN,), -1, dtype=ct.int32)
        if V6 == 1:
            kc = ct.load(kc_mask, index=idx, shape=(TN,)) > 0
            et = ct.load(elig, index=idx, shape=(TN,))
            elt = ct.load(elig_last, index=idx, shape=(TN,))
        for s in range(nsteps):
            t = t0 + s
            # 1. refractory countdown, then exact subthreshold integration
            rt = ct.maximum(rt - 1, 0)
            integ = rt == 0
            vn = rest_t + (vt - rest_t) * a + dt * (1.0 - a) + gt * coup - at * (k_adapt * (c - a))
            vt = ct.where(integ, vn, vt)
            gt = ct.where(integ, gt * b, gt)
            at = at * c
            # 2. threshold; each neuron spikes at most once per batch (refractory > DELAY),
            #    so the spike list is compacted once after the loop
            spike = integ & (vt > THRESHOLD)
            cnt = cnt + spike.astype(ct.int32)
            spike_at = ct.where(spike, t, spike_at)
            if V6 == 1:
                kcs = spike & kc
                at = ct.where(kcs, at + jump, at)
                decayed = et * ct.exp(-(t - elt).astype(ct.float32) * elig_scale).astype(ct.float64) + 1.0
                et = ct.where(kcs, decayed, et)
                elt = ct.where(kcs, ct.full((TN,), 0, dtype=ct.int64) + t, elt)
            # 3. arrivals emitted DELAY substeps ago; dropped while refractory
            slot = t & (RING - 1)
            raw = ct.reshape(ct.load(ginc, index=(slot, blk), shape=(1, TN)), (TN,))
            if RINGI == 1:
                inc = raw.astype(ct.float32) * unit
            else:
                inc = raw
            gt = ct.where(integ, gt + inc, gt)
            ct.store(ginc, index=(slot, blk), tile=ct.zeros((1, TN), dtype=ginc.dtype))
            # 4. reset this substep's spikers (also discards their arrival)
            vt = ct.where(spike, rest_t, vt)
            gt = ct.where(spike, 0.0, gt)
            rt = ct.where(spike, REFRACTORY, rt)
        spiked = spike_at >= 0
        m = spiked.astype(ct.int32)
        n = ct.sum(m)
        if n > 0:
            base = ct.atomic_add(log_counter, 0, n)
            pos = (base + ct.cumsum(m, 0) - m) & (LOG - 1)
            target = ct.where(spiked, pos, -1)
            ct.scatter(log_neuron, target, ids)
            ct.scatter(log_step, target, spike_at)
        ct.store(v, index=idx, tile=vt)
        ct.store(g, index=idx, tile=gt)
        ct.store(ref, index=idx, tile=rt)
        ct.store(adapt, index=idx, tile=at)
        if V6 == 1:
            ct.store(elig, index=idx, tile=et)
            ct.store(elig_last, index=idx, tile=elt)
        ct.store(counts, index=idx, tile=ct.load(counts, index=idx, shape=(TN,)) + cnt)

    @ct.kernel
    def deliver(log_neuron, log_step, batch_start, log_counter, ptr, post, weight, mod_mask, ginc,
                LOG: ct.Constant[int], G: ct.Constant[int], SPLIT: ct.Constant[int]):
        # Grid (G, SPLIT): CTAs along axis 0 stride over the batch's spikes; the SPLIT CTAs
        # along axis 1 share one spike's edge list so an 11k-edge hub does not serialise a batch.
        start = ct.load(batch_start, index=(0,), shape=(1,)).item()
        end = ct.load(log_counter, index=(0,), shape=(1,)).item()
        part = ct.bid(1)
        for k in range(start + ct.bid(0), end, G):
            pos = k & (LOG - 1)
            i = ct.gather(log_neuron, pos).item()
            t = ct.gather(log_step, pos).item()
            if ct.gather(mod_mask, i).item() == 0:
                slot = (t + DELAY) & (RING - 1)
                e0 = ct.gather(ptr, i).item() + part * TE
                e1 = ct.gather(ptr, i + 1).item()
                for e in range(e0, e1, SPLIT * TE):
                    eidx = e + ct.arange(TE, dtype=ct.int32)
                    j = ct.where(eidx < e1, ct.gather(post, eidx), -1)
                    w = ct.gather(weight, eidx)
                    ct.atomic_add(ginc, (slot, j), w)      # weight and ginc share a dtype (float32, or int32 fixed-point)


def build_record():
    record = {'model_revision': MODEL_REVISION, 'backend': 'cutile',
              'kernel_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'compile_flags': [], 'tile_neurons': TN, 'tile_edges': TE, 'ring_slots': RING, 'deliver_grid': DELIVER_GRID, 'deliver_split': DELIVER_SPLIT,
              'ring_mode': RING_MODE, 'ring_unit_mV': RING_UNIT if RING_INT else None}
    if AVAILABLE:
        try:
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            record.update({'cuda_tile': ct.__version__, 'cupy': cp.__version__,
                           'driver_version': int(cp.cuda.runtime.driverGetVersion()),
                           'device': props['name'].decode() if isinstance(props['name'], bytes) else str(props['name']),
                           'compute_capability': device.compute_capability})
        except Exception as error:  # pragma: no cover
            record['device_error'] = str(error)
    else:
        record['import_error'] = str(IMPORT_ERROR)
    return record


BUILD = build_record()


class _Device:
    """Device-side state; created on first use so construction needs no GPU."""


def _mirrored(name, device_attr, convert=None):
    """Host array property that is refreshed from the device lazily after each advance."""
    store = '_host_' + name

    def get(self):
        d = getattr(self, 'dev', None)
        if d is not None and name in self._stale:
            getattr(d, device_attr)[:self.n].get(out=self._pinned_view(device_attr), stream=self.stream)
            self.stream.synchronize()
            src = self._pinned_view(device_attr)
            getattr(self, store)[:] = src if convert is None else convert(src)
            self._stale.discard(name)
        return getattr(self, store)

    def set(self, value):
        setattr(self, store, value)
    return property(get, set)


class CuTileBrain(Brain):
    backend = 'cutile'
    v = _mirrored('v', 'v')
    g = _mirrored('g', 'g')
    refractory = _mirrored('refractory', 'ref')
    MIRRORED = ('v', 'g', 'refractory')

    def __init__(self, path, dt=.1):
        if not AVAILABLE:
            raise RuntimeError(f'cuTile backend unavailable: {IMPORT_ERROR}')
        self._stale = set()
        super().__init__(path, dt)
        self.previous_drive = np.zeros(self.n, dtype=np.float32)
        self.last = np.full(self.n, -1, dtype=np.int64)
        self.weights_dirty = True
        self.dev = None
        self.stream = cp.cuda.Stream(non_blocking=True)

    def _pinned_view(self, device_attr):
        return self.dev.pinned[device_attr][:self.n]

    # --- physiology hooks (the v6 mixin overrides these) -------------------
    def _physiology(self):
        n = self.n
        return {'rest': np.full(n, -52, np.float32), 'adaptation': np.zeros(n, np.float32),
                'kc_mask': np.zeros(n, np.uint8), 'modulation_mask': np.zeros(n, np.uint8),
                'eligibility': np.zeros(n, np.float64), 'eligibility_last': np.full(n, -1, np.int64),
                'adaptation_tau': 200.0, 'adaptation_jump': 0.0, 'eligibility_tau_ms': 1000.0, 'v6': 0}

    def mark_weights_changed(self):
        self.weights_dirty = True

    # --- device management -------------------------------------------------
    def _ensure_device(self):
        if self.dev is not None:
            return
        n = self.n
        d = _Device()
        d.grid = -(-n // TN)
        d.npad = d.grid * TN
        d.log = _log_capacity(d.npad)
        with self.stream:
            d.ptr = cp.asarray(self.ptr.astype(np.int32))
            d.post = cp.asarray(self.post)
            d.weight = cp.asarray(self.weight)
            for name, dtype in [('v', np.float32), ('g', np.float32), ('ref', np.int32), ('adapt', np.float32),
                                ('rest', np.float32), ('drive', np.float32), ('kc_mask', np.int32),
                                ('mod_mask', np.int32), ('counts', np.int32)]:
                setattr(d, name, cp.zeros(d.npad, dtype))
            d.elig = cp.zeros(d.npad, np.float64)
            d.elig_last = cp.zeros(d.npad, np.int64)
            d.ginc = cp.zeros((RING, d.npad), np.int32 if RING_INT else np.float32)
            d.weight_ring = d.weight if not RING_INT else cp.zeros(len(self.weight), np.int32)
            d.log_neuron = cp.zeros(d.log, np.int32)
            d.log_step = cp.full(d.log, -1, np.int32)
            d.log_counter = cp.zeros(1, np.int32)
            d.batch_start = cp.zeros(1, np.int32)
            d.retina = cp.asarray(self.retina); d.lamina = cp.asarray(self.lamina); d.sugar = cp.asarray(self.sugar)
            d.tonic = cp.zeros(d.npad, np.float32)
        # Pinned staging buffers: pageable copies of these arrays cost ~0.5 ms each.
        d.pinned = {}
        for name, dtype in [('drive', np.float32), ('counts', np.int32), ('v', np.float32), ('g', np.float32), ('ref', np.int32), ('adapt', np.float32), ('elig', np.float64), ('elig_last', np.int64)]:
            mem = cp.cuda.alloc_pinned_memory(d.npad * np.dtype(dtype).itemsize)
            d.pinned[name] = np.frombuffer(mem, dtype, d.npad)
        self.dev = d
        self.weights_dirty = False
        self.load_state_from_host()
        self._warm_up()

    def _constants(self):
        cached = getattr(self, '_constants_cache', None)
        if cached is not None:
            return cached
        dt = self.dt
        phys = self._physiology()
        a = float(np.float32(math.exp(-dt / 20)))
        b = float(np.float32(math.exp(-dt / 5)))
        tau = float(phys['adaptation_tau'])
        c = float(np.float32(math.exp(-dt / tau)))
        out = {'a': a, 'b': b, 'coup': float(np.float32((np.float32(a) - np.float32(b)) / np.float32(3))),
                'c': c, 'k_adapt': float(np.float32(tau / (tau - 20.0))), 'jump': float(phys['adaptation_jump']),
                'elig_scale': float(np.float32(dt / phys['eligibility_tau_ms'])), 'v6': int(phys['v6'])}
        self._constants_cache = out
        return out

    def _warm_up(self):
        """Compile both kernels before the first real substep (a zero-length batch)."""
        d = self.dev
        cst = self._constants()
        with self.stream:
            cp.copyto(d.batch_start, d.log_counter)
            self._launch_evolve(self.cursor, 0, cst)
            self._launch_deliver()
        self.stream.synchronize()

    def _launch_evolve(self, t0, nsteps, cst):
        d = self.dev
        ct.launch(self.stream, (d.grid,), evolve_batch,
                  (d.v, d.g, d.ref, d.adapt, d.rest, d.drive, d.kc_mask, d.elig, d.elig_last, d.ginc, d.counts,
                   d.log_neuron, d.log_step, d.log_counter, int(t0), int(nsteps), cst['a'], cst['b'], cst['coup'],
                   cst['c'], cst['k_adapt'], cst['jump'], cst['elig_scale'], d.log, cst['v6'], RING_INT, float(np.float32(RING_UNIT))))

    def _launch_deliver(self):
        d = self.dev
        ct.launch(self.stream, (DELIVER_GRID, DELIVER_SPLIT), deliver,
                  (d.log_neuron, d.log_step, d.batch_start, d.log_counter, d.ptr, d.post, d.weight_ring, d.mod_mask,
                   d.ginc, d.log, DELIVER_GRID, DELIVER_SPLIT))

    def _refresh_weight_ring(self, d, indices=None):
        """Keep the ring-side weights in sync with d.weight (fixed-point copy in int mode)."""
        if not RING_INT: return
        src = d.weight if indices is None else d.weight[indices]
        q = cp.rint(src.astype(np.float64) / RING_UNIT).astype(np.int32)
        if indices is None: d.weight_ring[:] = q
        else: d.weight_ring[indices] = q

    def load_state_from_host(self):
        """Upload host state (after a checkpoint restore or reset) and rebuild the ring."""
        self._ensure_device()
        d = self.dev
        n = self.n
        phys = self._physiology()
        with self.stream:
            self._stale.clear()
            d.v[:n].set(self.v); d.v[n:] = -52.0; d.g[:n].set(self.g); d.ref[:n].set(self.refractory.astype(np.int32))
            d.drive[:n].set(self.drive)
            d.rest[:n].set(phys['rest']); d.rest[n:] = -52.0
            d.adapt[:n].set(phys['adaptation']); d.kc_mask[:n].set(phys['kc_mask'].astype(np.int32))
            d.mod_mask[:n].set(phys['modulation_mask'].astype(np.int32))
            d.elig[:n].set(phys['eligibility']); d.elig_last[:n].set(phys['eligibility_last'])
            d.weight.set(self.weight); self.weights_dirty = False
            self._refresh_weight_ring(d)
            d.counts.fill(0); d.ginc.fill(0); d.log_counter.fill(0)
            # Pending CPU-format deliveries: slot s of the 19-slot queue holds spikes
            # delivered at the unique substep T in [cursor, cursor+17] with T%19==s.
            pending = []
            slots = self.queue.shape[0]
            for T in range(self.cursor, self.cursor + DELAY):
                s = T % slots
                count = int(self.queue_count[s])
                if count:
                    for i in self.queue[s, :count]:
                        pending.append((int(i), T - DELAY))
            if pending:
                neurons = np.asarray([p[0] for p in pending], np.int32)
                steps = np.asarray([p[1] for p in pending], np.int32)
                d.log_neuron[:len(pending)].set(neurons); d.log_step[:len(pending)].set(steps)
                d.batch_start.fill(0); d.log_counter.fill(len(pending))
                self._launch_deliver()
        self.stream.synchronize()

    # --- simulation ------------------------------------------------------
    def _device_drive(self, d):
        """Rebuild the device drive from the same recipe as the host array (same float32 operations)."""
        recipe = getattr(self, '_drive_recipe', None)
        if recipe is None:      # unknown provenance (tests writing brain.drive directly): upload the host array
            d.pinned['drive'][:self.n] = self.drive
            d.drive[:self.n].set(d.pinned['drive'][:self.n])
            return
        d.drive.fill(0)
        if len(self.lamina): d.drive[d.lamina] = np.float32(recipe['lamina_bias'])
        if len(self.retina): d.drive[d.retina] = cp.asarray(np.asarray(recipe['retina'], dtype=np.float32))
        if recipe.get('sugar') and len(self.sugar): d.drive[d.sugar] = np.float32(30)
        if recipe.get('tonic') is not None: d.drive[:self.n] += d.tonic[:self.n]
        for indices, amplitude in recipe.get('pulses', ()):
            ix = np.asarray(indices, dtype=np.int32)
            if not len(ix): continue
            amp = np.asarray(amplitude, dtype=np.float32)
            d.drive[cp.asarray(ix)] += (cp.asarray(amp) if amp.shape else float(amp))
        self._drive_recipe = None

    def _advance(self, steps):
        self._ensure_device()
        d = self.dev
        cst = self._constants()
        with self.stream:
            if self.weights_dirty:
                d.weight.set(self.weight); self.weights_dirty = False
                self._refresh_weight_ring(d)
            self._device_drive(d)
            d.counts.fill(0)
            t = self.cursor
            remaining = int(steps)
            while remaining > 0:
                batch = min(DELAY, remaining)
                cp.copyto(d.batch_start, d.log_counter)
                self._launch_evolve(t, batch, cst)
                self._launch_deliver()
                t += batch
                remaining -= batch
            self.cursor = t
            self._sync_to_host()

    def _sync_to_host(self):
        """Download spike counts now; other state is pulled lazily on access."""
        d = self.dev
        n = self.n
        d.counts.get(out=d.pinned['counts'], stream=self.stream)
        self.stream.synchronize()
        self.counts[:] = d.pinned['counts'][:n]
        self._stale.update(self.MIRRORED)
        logged = int(d.log_counter.get())
        if logged >= (1 << 30):
            d.log_counter.fill(logged % d.log)

    def step(self, luminance, duration_ms, sugar=False, lamina_bias=12.0):
        if len(luminance) != len(self.retina) or not np.all(np.isfinite(luminance)): raise ValueError('Invalid retinal input')
        if not math.isfinite(duration_ms) or not math.isfinite(lamina_bias): raise ValueError('Finite duration and current required')
        steps = int(round(duration_ms / self.dt))
        if steps < 1: raise ValueError('Duration too short')
        self.luminance += (1 - math.exp(-steps * self.dt / 10)) * (np.clip(luminance, 0, 1) - self.luminance)
        self.drive.fill(0); self.drive[self.lamina] = lamina_bias; self.drive[self.retina] = 30 * self.luminance / (.02 + self.luminance)
        if sugar: self.drive[self.sugar] = 30
        self._drive_recipe = {'lamina_bias': lamina_bias, 'retina': self.drive[self.retina].copy(), 'sugar': bool(sugar)}
        start = time.perf_counter()
        self._advance(steps)
        wall = time.perf_counter() - start
        self.total_spikes += int(self.counts.sum()); self.sim_ms += steps * self.dt
        return self.counts.copy(), wall

    # --- CPU-format state export (checkpoints) -----------------------------
    def materialize_host_state(self):
        """Fill the lazy-kernel bookkeeping arrays so the CPU checkpoint format restores exactly."""
        if self.dev is None:
            return
        d = self.dev
        self.stream.synchronize()
        n = self.n
        cursor = self.cursor
        self.previous_drive[:] = self.drive
        self.last[:] = cursor - 1
        # The log keeps every spike of the last DELAY substeps (see _log_capacity);
        # entries older than that, and the -1 initial fill, fall outside the window.
        neurons = d.log_neuron.get(); steps = d.log_step.get()
        keep = (steps >= cursor - DELAY) & (steps < cursor)
        self.queue.fill(0); self.queue_count.fill(0)
        slots = self.queue.shape[0]
        for s in np.unique(steps[keep]):
            members = np.sort(neurons[keep & (steps == s)])
            slot = int((s + DELAY) % slots)
            self.queue[slot, :len(members)] = members; self.queue_count[slot] = len(members)
        phys = self._physiology()
        gap = THRESHOLD - phys['rest']
        active = np.flatnonzero((self.v > THRESHOLD) | (self.drive > gap) | (self.drive + self.g > gap)).astype(np.int32)
        self.active.fill(0); self.active_flag.fill(0)
        self.active[:len(active)] = active; self.active_flag[active] = 1; self.nactive[0] = len(active)
