"""Run independent seed replicas in separate processes with a shared protocol.

Parallelism changes wall-clock scheduling, not the equations or game cadence.
Each process owns its full neural state, weights, game, and output directory.
"""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from .common import require_single_blas_thread,OUT,save_json


def main():
    require_single_blas_thread()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT/'parallel-survival')
    p.add_argument('--seeds',default='41031,41032,41033')
    p.add_argument('--eval-seeds',default='61031,61032,61033')
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--train-episodes',type=int,default=2)
    p.add_argument('--episode-seconds',type=float,default=30.)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    seeds=[int(s) for s in a.seeds.split(',')]
    if not 1<=a.jobs<=min(4,os.cpu_count() or 1):p.error('--jobs must be between 1 and min(4, available CPUs)')
    if len(seeds)!=len(set(seeds)):p.error('Training seeds must be unique')
    commands=[[sys.executable,'-m','doom_learning.run','--out',str(a.out/str(seed)),
        '--seeds',str(seed),'--eval-seeds',a.eval_seeds,'--train-episodes',str(a.train_episodes),
        '--episode-seconds',str(a.episode_seconds)] for seed in seeds]
    if a.dry_run:print(json.dumps({'jobs':a.jobs,'commands':commands},indent=2));return
    # Build once before starting workers to avoid concurrent binary creation.
    from .brain import build
    build();a.out.mkdir(parents=True,exist_ok=True)
    def worker(seed,command):
        with (a.out/f'{seed}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        return seed
    done=[]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures=[pool.submit(worker,s,c) for s,c in zip(seeds,commands)]
        for future in as_completed(futures):
            done.append(future.result());save_json(a.out/'progress.json',{'completed_seeds':done,'total_seeds':len(seeds)})
    print(json.dumps({'status':'complete','seeds':done,'jobs':a.jobs}))


if __name__=='__main__':main()
