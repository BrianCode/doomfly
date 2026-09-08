"""Compare the CPU kernel and the cuTile backend on the full graph with identical inputs.

Reports wall time per tic, total and per-superclass spike counts, the fraction
of neurons with identical per-tic counts, and state divergence. Float32 atomics
make the GPU sum order unspecified, so exact agreement is expected only for the
first tics; this is a measurement, not a pass/fail gate.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def run(brain,frames,tic_ms):
    counts=[];walls=[]
    for lum in frames:
        c,w=brain.step(lum,tic_ms);counts.append(c.copy());walls.append(w)
    return np.asarray(counts),np.asarray(walls)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--graph',default=str(ROOT/'outputs/doom/malecns_v1/graph.npz'))
    p.add_argument('--tics',type=int,default=100);p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--tic-ms',type=float,default=28.6);p.add_argument('--out',default='')
    p.add_argument('--skip-cpu',action='store_true')
    a=p.parse_args()
    from doom.cutile_brain import CuTileBrain,BUILD
    rng=np.random.default_rng(0)
    g=CuTileBrain(a.graph)
    frames=[rng.uniform(0,1,len(g.retina)).astype(np.float32) for _ in range(a.tics)]
    groups={k:np.flatnonzero(g.superclass==k) for k in np.unique(g.superclass)}
    t0=time.perf_counter();gc,gw=run(g,frames,a.tic_ms);gpu_total=time.perf_counter()-t0
    report={'tics':a.tics,'tic_ms':a.tic_ms,'build':BUILD,
      'gpu':{'ms_per_tic_mean':1e3*gw.mean(),'ms_per_tic_median':1e3*np.median(gw),'ms_per_tic_max':1e3*gw.max(),
             'wall_total_s':gpu_total,'spikes_total':int(gc.sum()),'spikes_per_tic':gc.sum(axis=1).tolist()}}
    if a.repeat>1:
        reps=[]
        for r in range(a.repeat-1):
            h=CuTileBrain(a.graph);hc,_=run(h,frames,a.tic_ms)
            first_diff=next((i for i in range(a.tics) if not np.array_equal(hc[i],gc[i])),None)
            reps.append({'first_tic_with_different_counts':first_diff,'total_spikes':int(hc.sum()),
                         'neurons_identical_all_tics':float(np.mean(np.all(hc==gc,axis=0)))})
        report['gpu_repeat']=reps
    if not a.skip_cpu:
        from doom.native import NativeBrain
        c=NativeBrain(a.graph)
        t0=time.perf_counter();cc,cw=run(c,frames,a.tic_ms);cpu_total=time.perf_counter()-t0
        per_tic_identical=[float(np.mean(cc[i]==gc[i])) for i in range(a.tics)]
        first_diff=next((i for i in range(a.tics) if not np.array_equal(cc[i],gc[i])),None)
        report['cpu']={'ms_per_tic_mean':1e3*cw.mean(),'ms_per_tic_median':1e3*np.median(cw),'ms_per_tic_max':1e3*cw.max(),
             'wall_total_s':cpu_total,'spikes_total':int(cc.sum()),'spikes_per_tic':cc.sum(axis=1).tolist()}
        report['agreement']={'first_tic_with_different_counts':first_diff,
             'fraction_neurons_identical_per_tic_min':min(per_tic_identical),'fraction_neurons_identical_per_tic_mean':float(np.mean(per_tic_identical)),
             'total_spikes_ratio_gpu_over_cpu':float(gc.sum()/max(1,cc.sum())),
             'max_abs_dv_mV_final':float(np.abs(g.v-c.v).max()),'max_abs_dg_mV_final':float(np.abs(g.g-c.g).max()),
             'superclass':{k:{'cpu':int(cc[:,idx].sum()),'gpu':int(gc[:,idx].sum())} for k,idx in groups.items()},
             'speedup':float(cw.mean()/gw.mean())}
    text=json.dumps(report,indent=1)
    if a.out:Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(text+'\n')
    brief={k:v for k,v in report.items() if k!='build'}
    for k in ('gpu','cpu'):
        if k in brief:brief[k]={kk:vv for kk,vv in brief[k].items() if kk!='spikes_per_tic'}
    print(json.dumps(brief,indent=1))

if __name__=='__main__':main()
