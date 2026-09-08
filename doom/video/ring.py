"""Bounded hand-off from the simulation thread to the video thread.

`offer` never blocks. When the consumer falls behind, the oldest pending frame
is evicted and recorded as a drop; drops become gaps in the archive and in the
frame-to-tick map, never duplicated or synthesized frames.
"""
import threading
from collections import deque


class FrameRing:
    def __init__(self, capacity=8):
        if capacity < 1: raise ValueError('Ring capacity must be positive')
        self.capacity = int(capacity)
        self._items = deque()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.offered = 0
        self.dropped = 0
        self.max_depth = 0
        self.dropped_ticks = []          # list of [first_tick, last_tick] ranges

    def offer(self, item):
        """Enqueue `item` (must expose `.tick`); returns the evicted item or None."""
        evicted = None
        with self._lock:
            self._items.append(item)
            self.offered += 1
            if len(self._items) > self.capacity:
                evicted = self._items.popleft()
                self.dropped += 1
                tick = getattr(evicted, 'tick', None)
                if tick is not None:
                    if self.dropped_ticks and self.dropped_ticks[-1][1] == tick - 1: self.dropped_ticks[-1][1] = tick
                    else: self.dropped_ticks.append([tick, tick])
            self.max_depth = max(self.max_depth, len(self._items))
            self._ready.set()
        return evicted

    def drain(self, timeout=None):
        """Return every pending item in order (waits up to `timeout` for the first)."""
        if not self._ready.wait(timeout): return []
        with self._lock:
            items = list(self._items)
            self._items.clear()
            self._ready.clear()
        return items

    def wake(self):
        self._ready.set()

    @property
    def depth(self):
        with self._lock: return len(self._items)

    def stats(self):
        with self._lock:
            return {'offered': self.offered, 'dropped': self.dropped, 'queue_depth': len(self._items),
                    'max_depth_seen': self.max_depth, 'capacity': self.capacity,
                    'dropped_tick_ranges': [list(r) for r in self.dropped_ticks[-100:]]}
