"""cuTile (GPU) execution of the v6 candidate physiology.

The mixin replaces only the compiled integration step of MemoryBrain with the
cuTile kernels in doom/cutile_brain.py (per-neuron rest, KC adaptation and
eligibility, modulatory cells that deliver no fast current). The centered rate
rule (rule.py), stimulation, calibration, provenance and checkpoint contents
are unchanged. Not tracked on the GPU: the `modulation` trace, which the C++
kernel writes but nothing reads; it is exported as zeros.
"""
import numpy as np
from doom.cutile_brain import CuTileBrain, AVAILABLE, IMPORT_ERROR, BUILD as CUTILE_BUILD
from .brain import MemoryBrain, PARAMETERS
from .visual import VisualMemoryBrain

if AVAILABLE:
    import cupy as cp


class CuTileMemoryMixin:
    backend = 'cutile'

    def __init__(self, *args, **kwargs):
        if not AVAILABLE:
            raise RuntimeError(f'cuTile backend unavailable: {IMPORT_ERROR}')
        super().__init__(*args, **kwargs)
        self.build = {**self.build, 'backend': 'cutile', 'cutile': CUTILE_BUILD}
        self.weights_dirty = True
        self.dev = None
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._plastic_dev = None

    # physiology exposed to the shared kernels
    def _physiology(self):
        return {'rest': self.rest, 'adaptation': self.adaptation, 'kc_mask': self.circuit['kc_mask'],
                'modulation_mask': self.modulation_mask, 'eligibility': self.eligibility,
                'eligibility_last': self.eligibility_last, 'adaptation_tau': self.adaptation_tau,
                'adaptation_jump': self.adaptation_jump, 'eligibility_tau_ms': PARAMETERS['trace_kc_seconds'] * 1000,
                'v6': 1}

    def _store_physiology(self, d):
        n = self.n
        self.adaptation[:] = cp.asnumpy(d.adapt[:n], stream=self.stream)
        self.eligibility[:] = cp.asnumpy(d.elig[:n], stream=self.stream)
        self.eligibility_last[:] = cp.asnumpy(d.elig_last[:n], stream=self.stream)
        self.stream.synchronize()

    def mark_weights_changed(self):
        self.weights_dirty = True

    # CuTileBrain machinery reused explicitly (the mixin is not a CuTileBrain subclass)
    _ensure_device = CuTileBrain._ensure_device
    _constants = CuTileBrain._constants
    _warm_up = CuTileBrain._warm_up
    _launch_evolve = CuTileBrain._launch_evolve
    _launch_deliver = CuTileBrain._launch_deliver
    _sync_to_host = CuTileBrain._sync_to_host
    materialize_host_state = CuTileBrain.materialize_host_state

    def load_state_from_host(self):
        CuTileBrain.load_state_from_host(self)
        self._plastic_dev = cp.asarray(self.circuit['edges'].astype(np.int64))

    def _advance(self, steps, learning=False):
        self._ensure_device()
        if not self.weights_dirty:
            # The rate rule rewrites only the plastic edges between bins.
            self.dev.weight[self._plastic_dev] = cp.asarray(self.weight[self.circuit['edges']])
        CuTileBrain._advance(self, steps)

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
