"""Per-frame metrics sidecar and per-run manifest for exported video."""
import json
import time
from pathlib import Path


class MetricsWriter:
    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.manifest_path = self.directory/'video-manifest.json'
        self._file = (self.directory/'video-metrics.jsonl').open('a', buffering=1 << 16)
        self.rows = 0
        self.events = 0
        self.write_manifest()

    def frame(self, record):
        self._file.write(json.dumps(record, separators=(',', ':'))+'\n')
        self.rows += 1

    def event(self, kind, tick, frame_index, detail=None):
        row = {'event': kind, 'tick': tick, 'frame_index': frame_index, 'recorded_at_ms': int(time.time()*1000)}
        if detail: row['detail'] = detail
        self._file.write(json.dumps(row, separators=(',', ':'))+'\n')
        self.events += 1

    def write_manifest(self, **updates):
        self.manifest.update(updates)
        self.manifest['updated_at_ms'] = int(time.time()*1000)
        tmp = self.manifest_path.with_suffix('.json.partial')
        tmp.write_text(json.dumps(self.manifest, indent=1)+'\n')
        tmp.replace(self.manifest_path)

    def flush(self):
        self._file.flush()

    def close(self, **updates):
        self._file.flush(); self._file.close()
        self.write_manifest(**updates)
