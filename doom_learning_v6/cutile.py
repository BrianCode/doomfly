"""cuTile (GPU) execution of the v6 candidate physiology.

The mixin replaces only the compiled integration step of MemoryBrain with the
cuTile kernels in doom/cutile_brain.py (per-neuron rest, KC adaptation and
eligibility, modulatory cells that deliver no fast current). The centered rate
rule (rule.py) is evaluated on the device with the same closed-form update, so a
whole game tic (three to five <=10 ms bins) is queued with a single download of
the spike counts at the end. Stimulation, calibration, provenance and
checkpoint contents are unchanged. Not tracked on the GPU: the `modulation`
trace, which the C++ kernel writes but nothing reads; it is exported as zeros.
"""
import math
import time
import numpy as np
from doom.cutile_brain import CuTileBrain, AVAILABLE, IMPORT_ERROR, BUILD as CUTILE_BUILD, DELAY, _mirrored
from .brain import MemoryBrain, PARAMETERS
from .visual import VisualMemoryBrain
from .rule import PARAMETERS as RULE

if AVAILABLE:
    import cupy as cp

RULE_FIELDS = ('rate_kc', 'rate_dan', 'memory_u', 'memory_w')


def _mirrored_small(name):
    """Small device-resident array (rule state) pulled to the host lazily."""
    store = '_host_' + name

    def get(self):
        d = getattr(self, 'dev', None)
        if d is not None and name in self._stale:
            getattr(self, store)[:] = getattr(d, name).get()
            self._stale.discard(name)
        return getattr(self, store)

    def set(self, value):
        setattr(self, store, value)
    return property(get, set)


def _weight_get(self):
    """Host weights are authoritative except for the plastic edges the device rule rewrites."""
    d = getattr(self, 'dev', None)
    if d is not None and 'weight' in self._stale:
        self._host_weight[self.circuit['edges']] = d.weight[d.plastic].get()
        self._stale.discard('weight')
    return self._host_weight


class CuTileMemoryMixin:
    backend = 'cutile'
    v = _mirrored('v', 'v')
    g = _mirrored('g', 'g')
    refractory = _mirrored('refractory', 'ref')
    adaptation = _mirrored('adaptation', 'adapt')
    eligibility = _mirrored('eligibility', 'elig')
    eligibility_last = _mirrored('eligibility_last', 'elig_last')
    rate_kc = _mirrored_small('rate_kc')
    rate_dan = _mirrored_small('rate_dan')
    memory_u = _mirrored_small('memory_u')
    memory_w = _mirrored_small('memory_w')
    weight = property(_weight_get, lambda self, value: setattr(self, '_host_weight', value))
    MIRRORED = ('v', 'g', 'refractory', 'adaptation', 'eligibility', 'eligibility_last', *RULE_FIELDS, 'weight')

    def __init__(self, *args, **kwargs):
        if not AVAILABLE:
            raise RuntimeError(f'cuTile backend unavailable: {IMPORT_ERROR}')
        self._stale = set()
        super().__init__(*args, **kwargs)
        self.build = {**self.build, 'backend': 'cutile', 'cutile': CUTILE_BUILD}
        self.weights_dirty = True
        self.dev = None
        self.stream = cp.cuda.Stream(non_blocking=True)

    # ---- physiology exposed to the shared kernels --------------------------
    def _physiology(self):
        return {'rest': self.rest, 'adaptation': self.adaptation, 'kc_mask': self.circuit['kc_mask'],
                'modulation_mask': self.modulation_mask, 'eligibility': self.eligibility,
                'eligibility_last': self.eligibility_last, 'adaptation_tau': self.adaptation_tau,
                'adaptation_jump': self.adaptation_jump, 'eligibility_tau_ms': PARAMETERS['trace_kc_seconds'] * 1000,
                'v6': 1}

    def mark_weights_changed(self):
        self.weights_dirty = True

    # CuTileBrain machinery reused explicitly (the mixin is not a CuTileBrain subclass)
    _ensure_device = CuTileBrain._ensure_device
    _pinned_view = CuTileBrain._pinned_view
    _device_drive = CuTileBrain._device_drive
    _refresh_weight_ring = CuTileBrain._refresh_weight_ring
    _constants = CuTileBrain._constants
    _warm_up = CuTileBrain._warm_up
    _launch_evolve = CuTileBrain._launch_evolve
    _launch_deliver = CuTileBrain._launch_deliver
    materialize_host_state = CuTileBrain.materialize_host_state

    def load_state_from_host(self):
        CuTileBrain.load_state_from_host(self)
        d = self.dev
        c = self.circuit
        with self.stream:
            d.plastic = cp.asarray(c['edges'].astype(np.int64))
            d.pre = cp.asarray(c['pre'].astype(np.int64))
            d.dan = cp.asarray(c['dan'].astype(np.int64))
            d.gain_t = cp.asarray(c['gain']).T                     # (plastic, dan) float32, as gain.T on the CPU
            d.baseline = cp.asarray(self.baseline_plastic)
            d.dan_baseline = cp.asarray(self.dan_baseline_hz)
            for name in RULE_FIELDS:
                setattr(d, name, cp.asarray(getattr(self, '_host_' + name)))
            d.counts_total = cp.zeros(d.npad, np.int32)
            d.tonic[:self.n].set(self.tonic)
        self.stream.synchronize()
        self._tonic_seen = self.tonic.copy()

    # ---- drive: same recipe as the host array, rebuilt on the device --------
    def _compute_drive(self, luminance, duration_ms, *, stimulation=None, lamina_bias=12.):
        steps = super()._compute_drive(luminance, duration_ms, stimulation=stimulation, lamina_bias=lamina_bias)
        pulses = [] if stimulation is None else (stimulation if isinstance(stimulation, list) else [stimulation])
        self._drive_recipe = {'lamina_bias': lamina_bias, 'retina': (30 * self.luminance / (.02 + self.luminance)).astype(np.float32),
                              'sugar': False, 'tonic': True,
                              'pulses': [(np.asarray(ix, dtype=np.int32), np.asarray(a, dtype=np.float32)) for ix, a in pulses]}
        return steps

    # ---- one <=10 ms bin: kernels then the rate rule, all on the stream -----
    def _launch_bin(self, steps):
        self._ensure_device()
        d = self.dev
        cst = self._constants()
        with self.stream:
            if self.weights_dirty:
                d.weight.set(self._host_weight); self.weights_dirty = False
                self._refresh_weight_ring(d)
            if not np.array_equal(self.tonic, self._tonic_seen):     # calibration edits tonic after construction
                d.tonic[:self.n].set(self.tonic); self._tonic_seen = self.tonic.copy()
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
            d.counts_total += d.counts
        self._stale.update(self.MIRRORED)

    def _rule_on_device(self, h, learning):
        """rule.advance with CuPy arrays: identical closed-form expressions in float64."""
        if not math.isfinite(h) or h <= 0 or h > .0100001: raise ValueError('Rate bins must be 0--10 ms')
        d = self.dev
        with self.stream:
            kc_hz = d.counts[d.pre].astype(np.float64) / h
            dan_hz = d.counts[d.dan].astype(np.float64) / h - d.dan_baseline
            ak = math.exp(-h / RULE['trace_kc_seconds']); ad = math.exp(-h / RULE['trace_dan_seconds'])
            kmid = d.rate_kc * math.sqrt(ak) + kc_hz * (1 - math.sqrt(ak))
            dmid = d.rate_dan * math.sqrt(ad) + dan_hz * (1 - math.sqrt(ad))
            d.rate_kc[:] = d.rate_kc * ak + kc_hz * (1 - ak); d.rate_dan[:] = d.rate_dan * ad + dan_hz * (1 - ad)
            if self.weights_frozen: return
            drive = self.eta * (kc_hz * (d.gain_t @ dmid) - (d.gain_t @ dan_hz) * kmid) if learning else cp.zeros_like(d.memory_u)
            tu = RULE['memory_decay_seconds']; tw = RULE['weight_filter_seconds']
            eu = math.exp(-h / tu); ew = math.exp(-h / tw); c = tu / (tu - tw) * (eu - ew)
            old_u = d.memory_u.copy()
            d.memory_u[:] = old_u * eu + drive * tu * (-math.expm1(-h / tu))
            d.memory_w[:] = d.memory_w * ew + old_u * c + drive * tu * (-math.expm1(-h / tw) - c)
            lo = RULE['minimum_fraction'] - 1; hi = RULE['maximum_fraction'] - 1
            cp.clip(d.memory_u, lo, hi, out=d.memory_u); cp.clip(d.memory_w, lo, hi, out=d.memory_w)
            d.weight[d.plastic] = (d.baseline * (1 + d.memory_w)).astype(np.float32)
            self._refresh_weight_ring(d, d.plastic)

    def _finish(self):
        """Download this call's spike counts (one transfer per call) and update the clocks."""
        d = self.dev
        n = self.n
        with self.stream:
            d.counts_total.get(out=d.pinned['counts'], stream=self.stream)
        self.stream.synchronize()
        self.counts[:] = d.pinned['counts'][:n]
        with self.stream:
            d.counts_total.fill(0)
        logged = int(d.log_counter.get())
        if logged >= (1 << 30):
            d.log_counter.fill(logged % d.log)
        self.total_spikes += int(self.counts.sum()); self.sim_ms = self.cursor * self.dt
        return self.counts.copy()

    def _neural_step(self, luminance, duration_ms, *, learning=False, stimulation=None, lamina_bias=12.):
        steps = self._compute_drive(luminance, duration_ms, stimulation=stimulation, lamina_bias=lamina_bias)
        start = time.perf_counter()
        self._launch_bin(steps)
        counts = self._finish()
        return counts, time.perf_counter() - start

    def step(self, luminance, duration_ms, *, learning=False, stimulation=None, lamina_bias=12., _defer=False):
        if not math.isfinite(duration_ms) or duration_ms <= 0: raise ValueError('Invalid duration')
        remaining = round(duration_ms / self.dt)
        if remaining < 1: raise ValueError('Duration too short')
        wall = 0.
        while remaining:
            ticks = min(100, remaining); interval = ticks * self.dt
            steps = self._compute_drive(luminance, interval, stimulation=stimulation, lamina_bias=lamina_bias)
            start = time.perf_counter()
            self._launch_bin(steps)
            self._rule_on_device(interval / 1000, learning)
            wall += time.perf_counter() - start
            remaining -= ticks
        if _defer:
            return None, wall
        start = time.perf_counter()
        counts = self._finish()
        return counts, wall + time.perf_counter() - start

    def rgb_step(self, frame, duration_ms, **kwargs):
        """VisualMemoryBrain.rgb_step with every <=10 ms chunk queued before one download."""
        if not hasattr(self, 'r8'): raise AttributeError('rgb_step needs the visual model')
        from doom.game import retinal_samples
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8: raise ValueError('RGB uint8 required')
        ticks = round(duration_ms / self.dt)
        if not math.isfinite(duration_ms) or ticks < 1: raise ValueError('Invalid duration')
        h, w = frame.shape[:2]
        x = np.minimum((self.r8_uv[:, 0] * (w - 1)).astype(int), w - 1)
        y = np.minimum((self.r8_uv[:, 1] * (h - 1)).astype(int), h - 1)
        values = frame[y, x, self.r8_channel].astype(np.float32) / 255
        values = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)
        extra = kwargs.pop('stimulation', None)
        base = [] if extra is None else list(extra) if isinstance(extra, list) else [extra]
        light = retinal_samples(frame, self.uv)
        wall = 0.
        while ticks:
            n = min(100, ticks)
            self.r8_light += (1 - math.exp(-n * self.dt / 10)) * (values - self.r8_light)
            pulses = base + [(self.r8, 30 * self.r8_light / (.02 + self.r8_light))]
            _, t = self.step(light, n * self.dt, stimulation=pulses, _defer=True, **kwargs)
            wall += t; ticks -= n
        start = time.perf_counter()
        counts = self._finish()
        return counts, wall + time.perf_counter() - start

    # ---- lifecycle ----------------------------------------------------------
    def reset(self, keep_memory=False):
        super().reset(keep_memory)
        self.weights_dirty = True
        if self.dev is not None:
            self.load_state_from_host()

    def checkpoint(self, path):
        self.materialize_host_state()
        self.modulation.fill(0); self.modulation_last.fill(0)
        super().checkpoint(path)

    def restore(self, path):
        super().restore(path)
        self.weights_dirty = True
        self.load_state_from_host()


class CuTileMemoryBrain(CuTileMemoryMixin, MemoryBrain):
    pass


class CuTileVisualMemoryBrain(CuTileMemoryMixin, VisualMemoryBrain):
    pass
