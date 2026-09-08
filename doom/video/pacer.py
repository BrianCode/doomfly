"""Wall-clock constant-frame-rate pacing for a live output.

The simulation produces one frame per game tic at whatever speed the host
allows. A live stream needs a fixed cadence, so the pacer emits the newest
produced frame at each output slot and counts how often it had to re-send the
previous frame (`held`) or skip produced frames that arrived between slots
(`skipped`). Both counters are shown to viewers so duplicated frames are never
mistaken for new neural states.
"""
import time


class LivePacer:
    MAX_CATCHUP_S = 5.0

    def __init__(self, fps=30, clock=time.monotonic):
        if fps <= 0: raise ValueError('fps must be positive')
        self.period = 1.0 / fps
        self.clock = clock
        self.next_slot = None
        self.latest = None
        self.latest_id = None
        self.shown_id = None
        self.held = 0
        self.skipped = 0
        self.shown = 0
        self.produced = 0
        self.slots_abandoned = 0

    def submit(self, frame, frame_id):
        """Register a newly produced frame (only the newest is kept)."""
        if self.latest is not None and self.latest_id != self.shown_id: self.skipped += 1
        self.latest = frame
        self.latest_id = frame_id
        self.produced += 1

    def due(self, now=None):
        now = self.clock() if now is None else now
        if self.next_slot is None: self.next_slot = now
        return now >= self.next_slot

    def take(self, now=None):
        """Return (frame, held_flag) for the current slot, or None when no frame exists yet."""
        now = self.clock() if now is None else now
        if self.latest is None: return None
        if self.next_slot is None: self.next_slot = now
        held = self.latest_id == self.shown_id
        if held: self.held += 1
        self.shown_id = self.latest_id
        self.shown += 1
        self.next_slot += self.period
        # After a long stall (checkpoint, host hiccup) catch up with held frames so the
        # constant-rate stream keeps wall time, but never burst more than MAX_CATCHUP_S.
        if self.next_slot < now - self.MAX_CATCHUP_S:
            self.slots_abandoned += int((now - self.next_slot) / self.period)
            self.next_slot = now
        return self.latest, held

    def seconds_until_due(self, now=None):
        now = self.clock() if now is None else now
        if self.next_slot is None: return 0.0
        return max(0.0, self.next_slot - now)

    def stats(self):
        return {'produced': self.produced, 'shown': self.shown, 'held': self.held, 'skipped': self.skipped, 'slots_abandoned': self.slots_abandoned}
