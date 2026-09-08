"""Video export service: archive MP4 (neural time), metrics sidecar and manifest.

The simulation thread hands one `VideoTick` per game tic to `VideoService.offer`
(non-blocking). A worker thread composites the frame with telemetry panels and
writes it to a hardware encoder through ffmpeg. Every composited frame is
described by one line in `video-metrics.jsonl` (frame index, tick, game state,
learning telemetry, source frame digest) so the archive can be joined back to
`audit.jsonl` on `(run_id, tick)`.
"""
import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .compose import Compositor, LAYOUT_VERSION
from .ffmpeg import FfmpegSink, build_archive_cmd, select_encoder, default_render_device
from .metrics import MetricsWriter
from .ring import FrameRing

ARCHIVE_FPS = 35            # one frame per game tic
SEGMENT_SECONDS = 300


@dataclass
class VideoTick:
    tick: int
    frame: Any                      # uint8[H, W, 3] RGB, private copy
    counts: Any                     # int32[N] spike counts of this tic
    game: dict
    action: dict
    neural_ms: float
    wall_s: float
    brain_step_ms: float
    sim_speed: float
    source_frame_sha256: str
    episode: int = 0
    learning: Optional[dict] = None
    readouts: Optional[list] = None
    episodes: Optional[list] = None
    run_id: str = ''
    study_id: str = ''
    phase: str = 'baseline'
    window_ms: float = 28.6
    extra: dict = field(default_factory=dict)


def ffmpeg_version(ffmpeg='ffmpeg'):
    try:
        out = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0] if out else None
    except (subprocess.SubprocessError, OSError):
        return None


class VideoService:
    def __init__(self, out_dir, *, run_id, study_id, phase, model_revision, group_names=(), groups=(), display=None,
                 size=(1280, 720), overlays=('hud', 'bars', 'raster', 'learning'), encoder='auto', device=None,
                 backend='numpy', ring_capacity=8, segment_seconds=SEGMENT_SECONDS, ffmpeg='ffmpeg', extra_manifest=None):
        self.out_dir = Path(out_dir)/run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.study_id, self.phase = run_id, study_id, phase
        self.width, self.height = int(size[0]), int(size[1])
        self.device = device or default_render_device()
        self.encoder = select_encoder(encoder, device=self.device, ffmpeg=ffmpeg)
        self.segment_seconds = int(segment_seconds)
        self.compositor = Compositor(self.width, self.height, group_names=group_names, groups=groups, display=display,
                                     phase=phase, overlays=overlays, backend=backend)
        self.ring = FrameRing(ring_capacity)
        frame_bytes = self.width * self.height * 3
        self.sink = FfmpegSink(lambda start: build_archive_cmd(self.encoder, width=self.width, height=self.height, fps=ARCHIVE_FPS,
                                                                out_dir=self.out_dir, device=self.device, segment_seconds=self.segment_seconds,
                                                                ffmpeg=ffmpeg, start_number=start // (self.segment_seconds * ARCHIVE_FPS)),
                               name='archive', frame_bytes=frame_bytes, log_path=self.out_dir/'ffmpeg-archive.log')
        manifest = {'run_id': run_id, 'study_id': study_id, 'phase': phase, 'model_revision': model_revision,
                    'layout_version': LAYOUT_VERSION, 'compositor_backend': backend, 'encoder': self.encoder, 'render_device': self.device,
                    'ffmpeg_version': ffmpeg_version(ffmpeg), 'canvas': [self.width, self.height], 'overlays': list(overlays),
                    'outputs': [{'kind': 'archive', 'pattern': 'archive-%05d.mp4', 'codec': 'h264', 'fps': ARCHIVE_FPS,
                                 'fps_contract': 'neural time: one frame per game tic, 35 f/s = 1x brain time; drops are gaps, never duplicates',
                                 'segment_seconds': self.segment_seconds, 'frames_per_segment': self.segment_seconds * ARCHIVE_FPS,
                                 'command': self.sink.stats()['command']}],
                    'statement': 'Overlay pixels are spectator rendering from telemetry; they are not neural input. Neural input is the raw '
                                 '640x480 RGB whose SHA-256 is source_frame_sha256 in audit.jsonl and video-metrics.jsonl.',
                    'started_at_ms': int(time.time() * 1000), 'segments': [], 'dropped_tick_ranges': [], 'frames': 0,
                    **(extra_manifest or {})}
        self.metrics = MetricsWriter(self.out_dir, manifest)
        self.frame_index = 0
        self.compose_ms = 0.0
        self.write_ms = 0.0
        self.last_game = None
        self.last_episode = None
        self.failed = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name='doom-video', daemon=True)
        self._thread.start()

    # ---- producer side (simulation thread) --------------------------------
    def offer(self, tick):
        self.ring.offer(tick)

    # ---- consumer side ----------------------------------------------------
    def _loop(self):
        try:
            while not self._stop.is_set() or self.ring.depth:
                for item in self.ring.drain(timeout=0.25):
                    self._process(item)
                if self.frame_index and self.frame_index % (ARCHIVE_FPS * 60) == 0:
                    self.metrics.write_manifest(frames=self.frame_index, dropped_tick_ranges=self.ring.stats()['dropped_tick_ranges'])
        except Exception as error:      # never propagate into the simulation
            self.failed = f'{type(error).__name__}: {error}'

    def _events(self, item):
        game = item.game or {}
        events = []
        if self.last_game is not None:
            if game.get('health', 0) < self.last_game.get('health', 0): events.append(('damage', {'health': game.get('health'), 'from': self.last_game.get('health')}))
            if game.get('kills', 0) > self.last_game.get('kills', 0): events.append(('kill', {'kills': game.get('kills')}))
            if game.get('finished') and not self.last_game.get('finished'): events.append(('round_end', {'episode': item.episode, 'survived_ticks': game.get('tick')}))
        self.last_game = game
        return events

    def _process(self, item):
        t0 = time.perf_counter()
        self.compositor.push_raster(item.counts, item.window_ms)
        canvas = self.compositor.render(item)
        t1 = time.perf_counter()
        ok = self.sink.write(canvas.tobytes())
        t2 = time.perf_counter()
        self.compose_ms = 0.9 * self.compose_ms + 0.1 * (t1 - t0) * 1e3
        self.write_ms = 0.9 * self.write_ms + 0.1 * (t2 - t1) * 1e3
        if not ok:
            self.metrics.event('encoder_failure', item.tick, self.frame_index, {'error': self.sink.last_error})
            return
        index = self.frame_index
        self.frame_index += 1
        segment_frames = self.segment_seconds * ARCHIVE_FPS
        learning = item.learning or None
        row = {'frame_index': index, 'archive_file': 'archive-%05d.mp4' % (index // segment_frames), 'archive_frame': index % segment_frames,
               'tick': item.tick, 'episode': item.episode, 'neural_ms': round(item.neural_ms, 3), 'wall_s': round(item.wall_s, 3),
               'sim_speed': round(item.sim_speed, 4), 'brain_step_ms': round(item.brain_step_ms, 3),
               'game': {k: item.game.get(k) for k in ('health', 'kills', 'ammo', 'enemies', 'tick', 'finished') if k in item.game},
               'action': item.action, 'source_frame_sha256': item.source_frame_sha256, 'held': False}
        if learning:
            row['learning'] = {k: learning.get(k) for k in ('changed_edges', 'mean_efficacy', 'minimum_efficacy', 'maximum_efficacy',
                                                              'mean_absolute_change', 'bound_edges', 'efficacy_histogram', 'damage_events',
                                                              'delivered_ms', 'stimulus_active', 'KC_spikes_last_tic', 'DAN_spikes_last_tic',
                                                              'MBON_spikes_last_tic') if k in learning}
        self.metrics.frame(row)
        for kind, detail in self._events(item):
            self.metrics.event(kind, item.tick, index, detail)

    # ---- reporting --------------------------------------------------------
    def stats(self):
        return {'encoder': self.encoder, 'render_device': self.device, 'frames': self.frame_index, 'compose_ms': round(self.compose_ms, 2),
                'write_ms': round(self.write_ms, 2), 'failed': self.failed, 'ring': self.ring.stats(), 'outputs': [self.sink.stats()],
                'out_dir': str(self.out_dir), 'contract': self.metrics.manifest['outputs'][0]['fps_contract']}

    def close(self, timeout=30):
        self._stop.set()
        self.ring.wake()
        self._thread.join(timeout=timeout)
        self.sink.close(timeout=timeout)
        segments = []
        for path in sorted(self.out_dir.glob('archive-*.mp4')):
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
            segments.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': h.hexdigest()})
        self.metrics.close(frames=self.frame_index, segments=segments, dropped_tick_ranges=self.ring.stats()['dropped_tick_ranges'],
                           ended_at_ms=int(time.time() * 1000), ended='error' if self.failed else 'closed', failure=self.failed,
                           outputs=[{**self.metrics.manifest['outputs'][0], **{k: v for k, v in self.sink.stats().items() if k != 'command'}}])
