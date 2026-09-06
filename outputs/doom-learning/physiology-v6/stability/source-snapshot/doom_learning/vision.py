"""Controlled visual stimuli through the exact baseline sensory path and graph.

This evaluates signal propagation and distinguishability, not calibrated fly
vision or learned gameplay. Every condition starts with identical neural state.
"""
import argparse, time
from .common import require_single_blas_thread, annotations, GRAPH, OUT, save_json, digest


def stimulus(name, tick, width=640, height=480):
    import numpy as np
    y, x = np.mgrid[:height, :width]
    if name == 'black': mask = np.zeros((height, width), dtype=bool)
    elif name == 'white': mask = np.ones((height, width), dtype=bool)
    elif name == 'left': mask = x < width // 2
    elif name == 'right': mask = x >= width // 2
    elif name == 'vertical': mask = (x // 40) % 2 == 0
    elif name == 'horizontal': mask = (y // 40) % 2 == 0
    elif name == 'motion_left': mask = ((x + tick * 8) // 40) % 2 == 0
    elif name == 'motion_right': mask = ((x - tick * 8) // 40) % 2 == 0
    else: raise ValueError(name)
    return np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)


def run(seconds=1.0, out=OUT / 'vision.json'):
    import numpy as np
    from doom.native import NativeBrain, BUILD
    from doom.game import retinal_samples
    from doom.engine import NeuralControls
    import json
    b = NativeBrain(GRAPH)
    a = annotations(b.ids)
    types = a.type.fillna('')
    manifest = json.loads((GRAPH.parent / 'manifest.json').read_text())
    groups = {name: np.flatnonzero(mask) for name, mask in {
        'receptors': types.eq('R1-R6'), 'lamina': types.isin(['L1','L2','L3','L5']),
        'T4': types.str.startswith('T4'), 'T5': types.str.startswith('T5'),
        'Kenyon': types.str.startswith('KC'), 'visual_KCg_d': types.eq('KCg-d'),
        'MBON11': types.eq('MBON11'), 'PPL101': types.eq('PPL101'),
        'descending': a.superclass.eq('descending_neuron'),
        'DNp20': types.eq('DNp20'), 'DNpe017': types.eq('DNpe017')}.items()}
    fields = ['v','g','refractory','drive','previous_drive','queue','queue_count',
              'counts','luminance','active','active_flag','nactive','last']
    initial = {k:getattr(b,k).copy() for k in fields}
    rows=[]; patterns={}; ticks=round(seconds*35)
    for name in ['black','white','left','right','vertical','horizontal','motion_left','motion_right']:
        for k,v in initial.items():getattr(b,k)[:] = v
        b.cursor=0; b.sim_ms=0.; b.total_spikes=0
        controls=NeuralControls(manifest['readouts'], mode='bci')
        total=np.zeros(b.n, dtype=np.int64); traces=[]; start=time.perf_counter()
        for tick in range(ticks):
            light=retinal_samples(stimulus(name,tick),b.uv)
            steps=round((tick+1)*10000/35)-b.cursor
            counts,wall=b.step(light,steps*.1)
            total+=counts; action=controls.decode(counts,steps*.0001)
            traces.append({'tick':tick,'neural_ms':b.sim_ms,'kernel_ms':wall*1000,
                'groups':{k:int(counts[ix].sum()) for k,ix in groups.items()},
                'action':{k:action[k] for k in ['turn','forward','attack']}})
        patterns[name]=total
        row={'stimulus':name,'brain_seconds':b.sim_ms/1000,'wall_seconds':time.perf_counter()-start,
             'spike_sha256':digest(total),'groups':{k:{'cells':len(ix),'spikes':int(total[ix].sum()),
             'active_cells':int(np.count_nonzero(total[ix]))} for k,ix in groups.items()},'trace':traces}
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
    kc=groups['Kenyon']; visual=groups['visual_KCg_d']
    report={'schema':1,'model_revision':BUILD['model_revision'],'full_graph':True,
        'neurons':b.n,'edges':len(b.weight),'conditions':rows,
        'pairwise':[{ 'a':x,'b':y,'different_KCs':int(np.count_nonzero(patterns[x][kc]!=patterns[y][kc])),
          'different_visual_KCs':int(np.count_nonzero(patterns[x][visual]!=patterns[y][visual]))}
          for x,y in [('left','right'),('vertical','horizontal'),('motion_left','motion_right'),('black','white')]],
        'visual_memory_input_active':any(r['groups']['visual_KCg_d']['spikes']>0 for r in rows),
        'validated_fly_vision':False,'gameplay_vision_advantage_demonstrated':False,
        'limits':'Deterministic stimulus assays on one reconstruction; no independent animals, physiological tuning or behavior validation.'}
    save_json(out,report)
    return report


if __name__=='__main__':
    require_single_blas_thread()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,default=1.)
    p.add_argument('--out',default=str(OUT/'vision.json'));args=p.parse_args()
    if args.seconds<=0: p.error('--seconds must be positive')
    run(args.seconds,args.out)
