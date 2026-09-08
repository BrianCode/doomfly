"""Frame compositor: game view plus telemetry panels, drawn from recorded data.

Layout version 1 (1280x720): the 640x480 game frame is enlarged 1.5x with
nearest-neighbour sampling (a spatial resize of one real frame, never a blend
between frames) on the left; a 320 px panel on the right shows the HUD, the
per-superclass spike bars, a scrolling raster of the fixed display sample, the
learning efficacy histogram and recent round survival; a status strip along
the bottom carries the phase label and the timing contract. Overlay pixels are
spectator rendering of telemetry; they are never neural input.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAYOUT_VERSION = 1
PANEL_WIDTH = 320
STRIP_HEIGHT = 40
RASTER_ROWS = 128
RASTER_COLUMNS = 160
UNVALIDATED = 'TRAINING / EXPERIMENTAL / UNVALIDATED'
COLORS = {'bg': (14, 14, 18), 'panel': (22, 22, 28), 'text': (230, 230, 230), 'dim': (140, 140, 150),
          'bar': (90, 170, 255), 'bar_hot': (255, 120, 60), 'hist': (120, 220, 140), 'survival': (200, 180, 80),
          'warn': (255, 200, 60), 'strip': (0, 0, 0)}


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # older Pillow without FreeType sizing
        return ImageFont.load_default()


class GlyphAtlas:
    """ASCII glyphs rendered once with Pillow, blitted per line with NumPy (about 10x faster than per-frame FreeType)."""

    def __init__(self, size):
        font = _font(size)
        self.height = 0
        widths = {}
        for code in range(32, 127):
            box = font.getbbox(chr(code))
            widths[code] = max(1, box[2])
            self.height = max(self.height, box[3])
        self.height += 2
        cell = max(widths.values()) + 2
        self.glyphs = {}
        for code in range(32, 127):
            image = Image.new('L', (cell, self.height), 0)
            ImageDraw.Draw(image).text((0, 0), chr(code), fill=255, font=font)
            advance = widths[code] + 1 if code != 32 else max(2, widths[ord('n')] // 2 + 1)
            self.glyphs[code] = np.asarray(image)[:, :advance].copy()

    def line(self, text):
        parts = [self.glyphs.get(ord(ch), self.glyphs[ord('?')]) for ch in text]
        return np.concatenate(parts, axis=1) if parts else np.zeros((self.height, 0), np.uint8)

    def draw(self, canvas, x, y, text, color):
        if not text or y >= canvas.shape[0] or y + self.height <= 0: return
        alpha = self.line(text)
        h = min(self.height, canvas.shape[0] - y)
        w = min(alpha.shape[1], canvas.shape[1] - x)
        if h <= 0 or w <= 0: return
        a = alpha[:h, :w, None].astype(np.uint16)
        region = canvas[y:y + h, x:x + w]
        col = np.asarray(color, dtype=np.uint16)
        region[:] = ((region.astype(np.uint16) * (255 - a) + col * a) // 255).astype(np.uint8)


class Compositor:
    def __init__(self, width=1280, height=720, *, group_names=(), groups=(), display=None, phase='baseline',
                 overlays=('hud', 'bars', 'raster', 'learning'), source_shape=(480, 640), backend='numpy'):
        if backend != 'numpy': raise ValueError('Only the numpy compositor is implemented in this version')
        self.width, self.height = int(width), int(height)
        self.overlays = set(overlays)
        self.phase = phase
        self.group_names = [str(g) for g in group_names]
        self.groups = [np.asarray(g, dtype=np.int64) for g in groups]
        self.display = None if display is None else np.asarray(display, dtype=np.int64)
        self.game_width = self.width - PANEL_WIDTH
        self.game_height = self.height
        self._maps_for = None
        self._set_source(source_shape)
        self.raster = np.zeros((RASTER_ROWS, RASTER_COLUMNS), dtype=np.float32)
        self.font = GlyphAtlas(14); self.small = GlyphAtlas(11); self.big = GlyphAtlas(18)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frames = 0

    def _set_source(self, shape):
        sh, sw = int(shape[0]), int(shape[1])
        if self._maps_for == (sh, sw): return
        self._ys = (np.arange(self.game_height) * sh // self.game_height).astype(np.int64)
        self._xs = (np.arange(self.game_width) * sw // self.game_width).astype(np.int64)
        self._maps_for = (sh, sw)

    # ---- helpers ----------------------------------------------------------
    def _bars(self, canvas, x0, y0, w, h, values, color, max_value=None):
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0: return
        top = max_value if max_value else max(float(values.max()), 1e-9)
        h = min(h, canvas.shape[0] - y0)
        if h <= 2: return
        slot = w / len(values)
        for i, val in enumerate(values):
            bh = int(round(min(1.0, val / top) * (h - 2)))
            x = x0 + int(i * slot); x1 = max(x + 1, x0 + int((i + 1) * slot) - 1)
            if bh > 0: canvas[y0 + h - bh:y0 + h, x:x1] = color

    def push_raster(self, counts, window_ms):
        if self.display is None or counts is None: return
        rates = counts[self.display].astype(np.float32) / max(window_ms, 1e-3) * 1000.0     # Hz
        column = np.clip(rates / 200.0, 0, 1)                                                  # 200 Hz = white, as in the UI
        self.raster = np.roll(self.raster, -1, axis=1)
        self.raster[:len(column), -1] = column[:RASTER_ROWS]

    # ---- rendering --------------------------------------------------------
    def render(self, tick, *, produced_fps=None, shown_fps=None, held=None, skipped=None, label=None):
        """Return the composited canvas (the buffer is reused; copy it if it must outlive the next call).

        All pixel work happens in NumPy first; text is drawn in a single Pillow pass at the end,
        so the frame costs one array->image->array round trip.
        """
        canvas = self.canvas
        frame = np.asarray(tick.frame)
        self._set_source(frame.shape[:2])
        canvas[:self.game_height, :self.game_width] = np.take(np.take(frame, self._ys, axis=0), self._xs, axis=1)
        px = self.game_width
        canvas[:, px:] = COLORS['panel']
        texts = []                     # (x, y, text, color, font)
        inner = PANEL_WIDTH - 20
        y = 8
        if 'hud' in self.overlays:
            g = tick.game or {}; a = tick.action or {}
            texts.append((px + 10, y, f'DOOMFLY  {self.phase.upper()}', COLORS['text'], self.big)); y += 26
            texts.append((px + 10, y, f'tick {tick.tick}  episode {tick.episode}  neural {tick.neural_ms/1000:.1f} s', COLORS['dim'], self.small)); y += 16
            texts.append((px + 10, y, f'health {g.get("health", "?")}  kills {g.get("kills", "?")}  ammo {g.get("ammo", "?")}  enemies {g.get("enemies", "?")}', COLORS['text'], self.font)); y += 20
            texts.append((px + 10, y, f'turn {a.get("turn", 0):+.2f}  fwd {a.get("forward", 0):.2f}  {"ATTACK" if a.get("attack") else ""}', COLORS['text'], self.font)); y += 20
            spikes = int(tick.counts.sum()) if tick.counts is not None else '?'
            texts.append((px + 10, y, f'spikes/tic {spikes}  kernel {tick.brain_step_ms:.1f} ms', COLORS['dim'], self.small)); y += 18
        if 'bars' in self.overlays and self.groups and tick.counts is not None:
            texts.append((px + 10, y, 'spikes per superclass (last tic)', COLORS['dim'], self.small)); y += 14
            sums = np.asarray([tick.counts[ix].sum() for ix in self.groups], dtype=np.float64)
            self._bars(canvas, px + 10, y, inner, 90, sums, COLORS['bar']); y += 96
        if 'raster' in self.overlays and self.display is not None:
            texts.append((px + 10, y, f'raster: {len(self.display)} sampled neurons, {RASTER_COLUMNS} frames', COLORS['dim'], self.small)); y += 14
            h = min(RASTER_ROWS, max(0, self.height - y))
            if h > 0:
                cols = (np.arange(inner) * RASTER_COLUMNS // inner).astype(np.int64)
                strip = (self.raster[:h, cols] * 255).astype(np.uint8)
                canvas[y:y + h, px + 10:px + 10 + inner] = strip[:, :, None]
            y += RASTER_ROWS + 6
        if 'learning' in self.overlays:
            learning = tick.learning
            if learning:
                texts.append((px + 10, y, f'plastic edges {learning.get("plastic_edges")}  changed {learning.get("changed_edges")}', COLORS['text'], self.small)); y += 14
                texts.append((px + 10, y, f'efficacy mean {learning.get("mean_efficacy", 0):.3f}  min {learning.get("minimum_efficacy", 0):.3f}  max {learning.get("maximum_efficacy", 0):.3f}', COLORS['dim'], self.small)); y += 14
                texts.append((px + 10, y, f'damage events {learning.get("damage_events")}  PPL101 pulse {"ON" if learning.get("stimulus_active") else "off"}  {learning.get("delivered_ms", 0)/1000:.1f} s delivered', COLORS['dim'], self.small)); y += 14
                rng = learning.get('histogram_range', [0.1, 2.0])
                texts.append((px + 10, y, f'efficacy histogram [{rng[0]}, {rng[1]}]', COLORS['dim'], self.small)); y += 14
                self._bars(canvas, px + 10, y, inner, 60, learning.get('efficacy_histogram') or [], COLORS['hist']); y += 64
                texts.append((px + 10, y, UNVALIDATED, COLORS['warn'], self.small)); y += 16
            else:
                texts.append((px + 10, y, 'NO LEARNING (fixed weights)', COLORS['dim'], self.small)); y += 16
            episodes = getattr(tick, 'episodes', None) or []
            if episodes:
                texts.append((px + 10, y, 'round survival, last 12 (s)', COLORS['dim'], self.small)); y += 14
                self._bars(canvas, px + 10, y, inner, 50, [e.get('tick', 0) / 35 for e in episodes[-12:]], COLORS['survival']); y += 54
        # status strip over the bottom of the game view (darkened band, then text)
        band = canvas[self.height - STRIP_HEIGHT:, :]
        band[:] = (band.astype(np.uint16) * 85 // 255).astype(np.uint8)
        if label is None:
            label = (f'NEURAL-TIME ARCHIVE | 35 f/s = 1x brain time | captured at {tick.sim_speed:.2f}x wall'
                     f' | run {str(tick.run_id)[:8]} | {UNVALIDATED if tick.learning else self.phase.upper()}')
        texts.append((10, self.height - STRIP_HEIGHT + 4, label, COLORS['text'], self.font))
        extra = []
        if produced_fps is not None: extra.append(f'PRODUCED {produced_fps:.1f} f/s')
        if shown_fps is not None: extra.append(f'SHOWN {shown_fps:.0f} f/s')
        if held is not None: extra.append(f'HELD {held}')
        if skipped is not None: extra.append(f'SKIPPED {skipped}')
        extra.append(f'TICK {tick.tick}')
        texts.append((10, self.height - STRIP_HEIGHT + 22, ' | '.join(extra), COLORS['dim'], self.small))
        for x, ty, text, color, font in texts:
            if 0 <= ty < self.height: font.draw(canvas, x, ty, text, color)
        self.frames += 1
        return canvas
