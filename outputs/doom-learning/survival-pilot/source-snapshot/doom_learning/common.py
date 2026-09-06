"""Shared full-connectome preparation and reproducible experiment records."""
from pathlib import Path
import hashlib, json, os

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / 'outputs/doom/malecns_v1/graph.npz'
OUT = ROOT / 'outputs/doom-learning'


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def annotations(ids):
    import pyarrow.feather as feather
    return feather.read_table(ROOT / 'connectome_data/malecns_v1/annotations.feather').to_pandas().set_index('bodyId').loc[ids]


def require_single_blas_thread():
    # Enforce the setting before importing NumPy in CLI entry points. Do not
    # silently reconfigure an already-running experiment or its thread pools.
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        raise SystemExit('Launch with OPENBLAS_NUM_THREADS=1 before importing NumPy.')
