"""Exploratory, counterbalanced visual conditioning through pixels only.

Protocol is saved before any outcomes. Distinct blue stripe orientations are
chosen as a simple visual assay, never from Doom scores. Endogenous DAN firing
is allowed; paired effects must exceed no-imposed-US and backward effects.
This is not a replication of an odor experiment in a living fly.
"""
import argparse,time
import numpy as np
from .visual import VisualMemoryBrain
from doom_learning_v2.vision import frame_for
from doom_learning.common import OUT,save_json,digest


def run(out,eta=.05):
    from pathlib import Path
    out=Path(out)
    if out.exists():raise ValueError('Fresh output path required')
    out.mkdir(parents=True)
    save_json(out/'protocol.json',{'model':'kc-adaptive-lif-v4','eta':eta,'cue_names':['vertical','horizontal'],
        'counterbalance':[0,1],'conditions':['paired','backward','frozen','no_imposed_US'],
        'CS_ms':1000,'US_pulses_ms':[200,700,1200,1700],'US_current':200,'US_pulse_width_ms':1,
        'evaluation':'1.4 s including 0.4 s after cue offset; plasticity off; identical reset fast state.',
        'criterion':'Both cues must evoke MBON spikes. Counterbalanced paired cue-specific suppression must exceed no-imposed-US and backward controls; frozen responses identical. Otherwise fail.',
        'calibration':'eta is an exploratory conditioning parameter. No game scores used; this assay is development data, not independent confirmation.',
        'anatomy':'Full retained MaleCNS graph; only existing KC-to-MBON11 weights plastic.'})
    b=VisualMemoryBrain(eta=eta);frames=[frame_for(x) for x in ['vertical','horizontal']];black=frame_for('black')
    def advance(frame,start,end,learning=False,stim=None):
        total=np.zeros(b.n,dtype=np.int64)
        # Millisecond CS/US boundaries are exact; all long spans still sample
        # the photoreceptor filter at <= one 35-Hz game tic.
        at=start
        while at<end:
            dt=min(20,end-at);c,_=b.rgb_step(frame,dt,learning=learning,stimulation=stim);total+=c;at+=dt
        return total
    def probe(frame):
        b.reset(keep_memory=True);c=advance(frame,0,1000);c+=advance(black,1000,1400)
        return {'MBON':c[b.circuit['mb']].tolist(),'KC':int(c[b.circuit['kc']].sum()),'DAN':c[b.circuit['dan']].tolist(),'KC_pattern_sha256':digest(c[b.circuit['kc']])}
    rows=[]
    for plus in [0,1]:
        for condition in ['paired','backward','frozen','no_imposed_US']:
            b.reset();before=[probe(f) for f in frames];b.reset();start=time.perf_counter()
            cs_start=2000 if condition=='backward' else 0;cs_end=cs_start+1000
            pulses=[0,500,1000,1500] if condition=='backward' else [200,700,1200,1700]
            if condition=='no_imposed_US':pulses=[]
            duration=max(1800,cs_end)
            bounds=sorted(set([0,cs_start,cs_end,duration,*pulses,*[p+1 for p in pulses]]));train=np.zeros(b.n,dtype=np.int64)
            for begin,end in zip(bounds,bounds[1:]):
                train+=advance(frames[plus] if cs_start<=begin<cs_end else black,begin,end,
                    learning=condition!='frozen',stim=(b.circuit['dan'],200.) if begin in pulses else None)
            memory=b.memory();after=[probe(f) for f in frames]
            suppression=[1-sum(y['MBON'])/sum(x['MBON']) if sum(x['MBON']) else None for x,y in zip(before,after)]
            row={'plus':plus,'condition':condition,'before':before,'after':after,'suppression':suppression,
                'selectivity':suppression[plus]-suppression[1-plus] if all(x is not None for x in suppression) else None,
                'memory':memory,'train_KC':int(train[b.circuit['kc']].sum()),'train_DAN':train[b.circuit['dan']].tolist(),'wall':time.perf_counter()-start}
            rows.append(row);save_json(out/f'{plus}-{condition}.json',row);print(row,flush=True)
    passed=True
    for plus in [0,1]:
        r={r['condition']:r for r in rows if r['plus']==plus};p=r['paired']['selectivity']
        passed &= p is not None and p>0 and all(r[x]['selectivity'] is not None and p>r[x]['selectivity'] for x in ['backward','no_imposed_US']) and r['frozen']['before']==r['frozen']['after']
    save_json(out/'results.json',{'complete':True,'qualitative_development_gate':bool(passed),'independently_validated':False,'rows':rows})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT/'physiology-v4/rgb-conditioning'));p.add_argument('--eta',type=float,default=.05);args=p.parse_args();run(args.out,args.eta)
