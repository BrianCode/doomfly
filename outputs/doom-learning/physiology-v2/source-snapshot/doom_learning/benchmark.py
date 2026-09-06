"""Measured headless throughput with independent replicas and replay checks.

This changes execution scheduling only. Both batches use the same game seeds,
0.1 ms neural integrator and fixed control mapping. Setup time is reported.
"""
import argparse,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .common import OUT,require_single_blas_thread,save_json


def worker(out,seed,seconds):
    from .brain import MemoryBrain
    from .run import episode
    import hashlib
    start=time.perf_counter();b=MemoryBrain();setup=time.perf_counter()-start
    result,_=episode(b,seed,seconds,learning=False)
    # Include measured inputs, spikes, actions, health and memory, but no timing.
    signature=hashlib.sha256(json.dumps(result['trace'],sort_keys=True).encode()).hexdigest()
    save_json(out,{'seed':seed,'trace_sha256':signature,'setup_seconds':setup,'timing':result['timing']})


def main():
    require_single_blas_thread();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT/'headless-benchmark');p.add_argument('--seconds',type=float,default=1.)
    p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int,default=81031);a=p.parse_args()
    if a.seconds<=0:p.error('Positive duration required')
    if a.worker:worker(a.out,a.seed,a.seconds);return
    from .brain import build
    build();a.out.mkdir(parents=True,exist_ok=True);batches=[]
    for batch,jobs in enumerate([1,2,2,1]):
        def run(seed):
            out=a.out/f'batch-{batch}-jobs-{jobs}-seed-{seed}.json'
            subprocess.run([sys.executable,'-m','doom_learning.benchmark','--worker','--seed',str(seed),
                            '--seconds',str(a.seconds),'--out',str(out)],check=True)
            return json.loads(out.read_text())
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=jobs) as pool:rows=list(pool.map(run,[81031,81032]))
        wall=time.perf_counter()-start
        batches.append({'jobs':jobs,'batch_wall_seconds':wall,'aggregate_brain_seconds_per_wall_second':
                        sum(r['timing']['neural_seconds'] for r in rows)/wall,'runs':rows})
    equal=all(x['trace_sha256']==y['trace_sha256'] for batch in batches[1:] for x,y in zip(batches[0]['runs'],batch['runs']))
    report={'schema':1,'batches':batches,'identical_traces_across_scheduling':equal,
            'limits':'Short full-graph benchmark including process/graph setup, on a shared host. Not a steady-state hardware benchmark or evidence of learning. Parallel jobs require independent full neural and game state.'}
    save_json(a.out/'results.json',report);print(json.dumps(report),flush=True)
    if not equal:raise SystemExit('Scheduling comparison failed: traces differ')


if __name__=='__main__':main()
