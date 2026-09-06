"""Probe response, recruitment, and post-stimulus recovery before conditioning."""
import argparse,time
from pathlib import Path
import numpy as np
from .calibration import calibrated_brain
from doom_learning_v2.vision import frame_for
from doom_learning.common import save_json,capture_provenance


def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required')
    out.mkdir(parents=True);capture_provenance(out,additional=['doom_learning_v2','doom_learning_v6'])
    save_json(out/'protocol.json',{'warmup_seconds':2,'cue_seconds':1,'recovery_seconds':3,
        'cues':['black','left_blue','right_blue','white'],
        'learning':False,'weights':'frozen','interpretation':'Physiological diagnostic, not a conditioning or survival test. No parameters fit to these outcomes.'})
    b=calibrated_brain();b.weights_frozen=True;rows=[];dark=frame_for('black')
    for cue in ['black','left_blue','right_blue','white']:
        b.reset();trace=[];start=time.perf_counter();frame=frame_for(cue)
        for t in range(600):
            c,_=b.rgb_step(frame if 200<=t<300 else dark,10,learning=False)
            trace.append({'ms':b.sim_ms,'KC':int(c[b.circuit['kc']].sum()),
                'active_KC':int(np.count_nonzero(c[b.circuit['kc']])),
                'DAN':c[b.circuit['dan']].tolist(),'MBON':c[b.circuit['mb']].tolist(),'all_spikes':int(c.sum())})
        def window(a,z):
            r=trace[a:z];s=(z-a)/100
            return {'KC_hz_total':sum(x['KC'] for x in r)/s,
                    'DAN_hz':[sum(x['DAN'][i] for x in r)/s for i in range(2)],
                    'MBON_hz':[sum(x['MBON'][i] for x in r)/s for i in range(2)],
                    'all_spikes_hz':sum(x['all_spikes'] for x in r)/s}
        row={'cue':cue,'baseline':window(100,200),'during':window(200,300),'recovery':window(500,600),'wall_seconds':time.perf_counter()-start,'trace':trace}
        save_json(out/f'{cue}.json',row);rows.append({k:v for k,v in row.items() if k!='trace'});print(rows[-1],flush=True)
    save_json(out/'results.json',{'complete':True,'rows':rows,'survival_learning_demonstrated':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/doom-learning/physiology-v6/stability');a=p.parse_args();run(a.out)
