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


def capture_provenance(out,additional=()):
    """Freeze exact sources and inputs before a new experiment starts."""
    import importlib.metadata, platform, shutil
    out=Path(out);snapshot=out/'source-snapshot';snapshot.mkdir(parents=True)
    sources={}
    for folder in ['doom_learning','doom',*additional]:
        for p in sorted((ROOT/folder).iterdir()):
            if p.suffix not in ['.py','.cpp']:continue
            relative=p.relative_to(ROOT);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target);sources[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    h=hashlib.sha256()
    with GRAPH.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    record={'schema':2,'captured_before_first_episode':True,'python':platform.python_version(),
            'platform':platform.platform(),'graph_sha256':h.hexdigest(),'source_sha256':sources,
            'packages':{k:importlib.metadata.version(k) for k in ['numpy','vizdoom','numba','Pillow','pyarrow']},
            'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS')}
    save_json(out/'provenance.json',record);return record
