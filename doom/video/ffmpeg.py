"""ffmpeg encoder sinks: hardware H.264 through VAAPI (Intel iGPU), QSV, NVENC or libx264.

Frames are written as raw rgb24 to ffmpeg's stdin; ffmpeg converts to NV12,
encodes on the selected engine and muxes. The archive sink writes fragmented
MP4 segments in neural time (one frame per game tic); the live sink (used
later) writes a constant-rate stream. Stream keys never appear in the logged
command: `redacted_command()` masks the RTMP path.
"""
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

ENCODER_ORDER = ['vaapi', 'qsv', 'nvenc', 'x264']
INTEL_RENDER_LINK = Path('/dev/dri/by-path/pci-0000:00:02.0-render')


def default_render_device():
    if INTEL_RENDER_LINK.exists():
        return str(INTEL_RENDER_LINK.resolve())
    for candidate in sorted(Path('/dev/dri').glob('renderD*')) if Path('/dev/dri').exists() else []:
        return str(candidate)
    return '/dev/dri/renderD128'


def encoder_args(encoder, *, device, rate_control):
    """Input-side hardware options and the codec options for one encoder.

    rate_control: 'archive' (quality-targeted VBR) or 'live' (CBR for ingest).
    Returns (pre_input_args, codec_args).
    """
    if encoder == 'vaapi':
        pre = ['-vaapi_device', device]
        vf = ['-vf', 'format=nv12,hwupload']
        rc = ['-rc_mode', 'VBR', '-b:v', '6M', '-maxrate', '12M', '-g', '70', '-bf', '2'] if rate_control == 'archive' \
            else ['-rc_mode', 'CBR', '-b:v', '4500k', '-maxrate', '4500k', '-bufsize', '9000k', '-g', '60', '-bf', '0']
        return pre, vf + ['-c:v', 'h264_vaapi', '-profile:v', 'high'] + rc
    if encoder == 'qsv':
        pre = ['-init_hw_device', f'qsv=hw,child_device={device}', '-filter_hw_device', 'hw']
        vf = ['-vf', 'format=nv12,hwupload=extra_hw_frames=64']
        rc = ['-preset', 'medium', '-b:v', '6M', '-maxrate', '12M', '-look_ahead', '1', '-g', '70', '-bf', '2'] if rate_control == 'archive' \
            else ['-preset', 'veryfast', '-b:v', '4500k', '-maxrate', '4500k', '-bufsize', '9000k', '-g', '60', '-bf', '0']
        return pre, vf + ['-c:v', 'h264_qsv', '-profile:v', 'high'] + rc
    if encoder == 'nvenc':
        vf = ['-vf', 'format=nv12']
        rc = ['-preset', 'p5', '-rc', 'vbr', '-cq', '23', '-b:v', '0', '-maxrate', '12M', '-bufsize', '24M', '-g', '70', '-bf', '2',
              '-spatial-aq', '1', '-temporal-aq', '1'] if rate_control == 'archive' \
            else ['-preset', 'p4', '-tune', 'll', '-rc', 'cbr', '-b:v', '4500k', '-maxrate', '4500k', '-bufsize', '9000k', '-g', '60', '-bf', '0',
                  '-forced-idr', '1', '-no-scenecut', '1', '-rc-lookahead', '0', '-spatial-aq', '1']
        return [], vf + ['-c:v', 'h264_nvenc', '-profile:v', 'high', '-pix_fmt', 'yuv420p'] + rc
    if encoder == 'x264':
        rc = ['-preset', 'veryfast', '-crf', '23', '-g', '70'] if rate_control == 'archive' \
            else ['-preset', 'veryfast', '-tune', 'zerolatency', '-b:v', '4500k', '-maxrate', '4500k', '-bufsize', '9000k', '-g', '60']
        return [], ['-c:v', 'libx264', '-pix_fmt', 'yuv420p'] + rc
    raise ValueError(f'Unknown encoder {encoder!r}')


def build_archive_cmd(encoder, *, width, height, fps, out_dir, device=None, segment_seconds=300, ffmpeg='ffmpeg', start_number=0):
    """One frame per game tic, pts in neural time, fragmented MP4 segments."""
    device = device or default_render_device()
    pre, codec = encoder_args(encoder, device=device, rate_control='archive')
    return [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-nostats', '-progress', 'pipe:1', *pre,
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0',
            *codec, '-force_key_frames', f'expr:gte(t,n_forced*{segment_seconds})',
            '-f', 'segment', '-segment_time', str(segment_seconds), '-reset_timestamps', '1',
            '-segment_start_number', str(start_number), '-segment_format', 'mp4',
            '-segment_format_options', 'movflags=+frag_keyframe+empty_moov+default_base_moof',
            str(Path(out_dir)/'archive-%05d.mp4')]


def build_live_cmd(encoder, *, width, height, fps, rtmp_url=None, hls_dir=None, device=None, ffmpeg='ffmpeg', hls_seconds=2):
    """Constant-rate live encode with silent audio, tee'd to RTMP and/or HLS."""
    if not rtmp_url and not hls_dir: raise ValueError('A live sink needs an RTMP URL or an HLS directory')
    device = device or default_render_device()
    pre, codec = encoder_args(encoder, device=device, rate_control='live')
    outputs = []
    if rtmp_url: outputs.append(f'[f=flv:onfail=ignore]{rtmp_url}')
    if hls_dir:
        outputs.append('[f=hls:hls_time=%d:hls_list_size=6:hls_flags=delete_segments+independent_segments+temp_file+program_date_time'
                       ':hls_segment_type=fmp4:hls_fmp4_init_filename=init.mp4]%s' % (hls_seconds, Path(hls_dir)/'live.m3u8'))
    return [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-nostats', '-progress', 'pipe:1', *pre,
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0',
            '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', *codec, '-c:a', 'aac', '-b:a', '64k', '-shortest',
            '-f', 'tee', '-use_fifo', '1', '-fifo_options', 'drop_pkts_on_overflow=1:attempt_recovery=1:recovery_wait_time=1',
            '|'.join(outputs)]


def redacted_command(cmd):
    return [re.sub(r'(rtmps?://[^/]+/[^/]+/)[^|\s\]]+', r'\1<redacted>', part) for part in cmd]


def probe_encoders(candidates=ENCODER_ORDER, *, device=None, ffmpeg='ffmpeg', timeout=20):
    """Return the candidate encoders that complete a 10-frame test encode on this host."""
    if not shutil.which(ffmpeg): return []
    device = device or default_render_device()
    working = []
    for encoder in candidates:
        try:
            pre, codec = encoder_args(encoder, device=device, rate_control='archive')
            cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', *pre, '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=30',
                   '-frames:v', '10', *codec, '-f', 'null', '-']
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if result.returncode == 0: working.append(encoder)
        except (subprocess.SubprocessError, OSError):
            continue
    return working


def select_encoder(requested='auto', **probe_kwargs):
    if requested != 'auto':
        return requested
    working = probe_encoders(**probe_kwargs)
    if not working: raise RuntimeError('No working video encoder found (ffmpeg missing?)')
    return working[0]


class FfmpegSink:
    """One ffmpeg process fed with raw frames; restarts itself if ffmpeg dies."""

    def __init__(self, cmd_builder, *, name, frame_bytes, log_path=None, max_restarts=5):
        self.cmd_builder = cmd_builder      # callable(start_number) -> argv
        self.name = name
        self.frame_bytes = frame_bytes
        self.log_path = Path(log_path) if log_path else None
        self.max_restarts = max_restarts
        self.process = None
        self.frames_written = 0
        self.restarts = 0
        self.last_error = None
        self.progress = {}
        self._progress_thread = None
        self._log = None
        self.started_at = None
        self.command = None
        self.spawn()

    def spawn(self):
        self.command = self.cmd_builder(self.frames_written)
        self._log = self.log_path.open('ab') if self.log_path else subprocess.DEVNULL
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                        bufsize=self.frame_bytes * 4)
        self.started_at = time.time()
        self._progress_thread = threading.Thread(target=self._read_progress, args=(self.process,), daemon=True)
        self._progress_thread.start()

    def _read_progress(self, process):
        try:
            for raw in process.stdout:
                line = raw.decode(errors='replace').strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    self.progress[key] = value
        except Exception:
            pass

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def write(self, frame):
        """Write one rgb24 frame (bytes or a C-contiguous uint8 array). Returns False on failure."""
        data = frame if isinstance(frame, (bytes, bytearray, memoryview)) else memoryview(frame)
        if len(data) != self.frame_bytes: raise ValueError('Frame size does not match the encoder configuration')
        if not self.alive:
            if self.restarts >= self.max_restarts:
                return False
            self.restarts += 1
            self.last_error = f'ffmpeg exited with {self.process.returncode}'
            self.spawn()
        try:
            self.process.stdin.write(data)
            self.frames_written += 1
            return True
        except (BrokenPipeError, OSError) as error:
            self.last_error = str(error)
            return False

    def stats(self):
        return {'name': self.name, 'frames_written': self.frames_written, 'restarts': self.restarts, 'alive': self.alive,
                'last_error': self.last_error, 'encoder_fps': self.progress.get('fps'), 'bitrate': self.progress.get('bitrate'),
                'command': redacted_command(self.command or [])}

    def close(self, timeout=30):
        if self.process is None: return
        try:
            if self.process.stdin: self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.last_error = 'ffmpeg killed after timeout on close'
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        self.process = None
