"""Counterbalanced RGB conditioning with physiological background activity."""
import argparse,time
from pathlib import Path
import numpy as np
from .calibration import calibrated_brain
from .brain import PARAMETERS
from doom_learning_v2.vision import frame_for
from doom_learning.common import OUT,save_json,capture_provenance,digest


def run(out,eta=.001,us_current=4.):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output path required')
    out.mkdir(parents=True);capture_provenance(out,additional=['doom_learning_v2','doom_learning_v5'])
    protocol={'model':'centered-antihebbian-v5','eta':eta,'parameters':PARAMETERS,
        'cue_names':['left_blue','right_blue'],'conditions':['paired','backward','frozen','no_imposed_US'],
        'warmup_ms':2000,'CS_ms':1000,'US_current':us_current,'US_width_ms':200,
        'forward_US_starts_ms':[200,700,1200,1700],'backward_US_starts_ms':[0,500,1000,1500],
        'backward_CS_start_ms':2200,'post_training_dark_ms':2000,
        'evaluation':'Same 2 s warmup followed by 1 s cue and 0.4 s dark; all efficacies held fixed during tests.',
        'development_gate':'Both counterbalances require positive paired cue-selectivity exceeding backward and no-US; frozen and memory reset responses must equal pre-training.',
        'status':'Development assay, not independent replication and not a survival experiment.'}
    save_json(out/'protocol.json',protocol)
    b=calibrated_brain(eta);save_json(out/'calibration.json',b.calibration)
    frames=[frame_for('left_blue'),frame_for('right_blue')];dark=frame_for('black')
    def span(frame,milliseconds,learning=False,stim=None):
        total=np.zeros(b.n,dtype=np.int64);trace=[]
        for _ in range(round(milliseconds/10)):
            c,_=b.rgb_step(frame,10,learning=learning,stimulation=stim);total+=c
            trace.append({'ms':b.sim_ms,'KC':int(c[b.circuit['kc']].sum()),'DAN':c[b.circuit['dan']].tolist(),'MBON':c[b.circuit['mb']].tolist()})
        return total,trace
    def warm():span(dark,2000)
    def probe(frame):
        b.reset(keep_memory=True);b.weights_frozen=True;warm()
        c,trace=span(frame,1000);d,tail=span(dark,400);c+=d
        return {'MBON':c[b.circuit['mb']].tolist(),'DAN':c[b.circuit['dan']].tolist(),
            'KC':int(c[b.circuit['kc']].sum()),'KC_pattern_sha256':digest(c[b.circuit['kc']]),'trace':trace+tail}
    rows=[]
    for plus in [0,1]:
        for condition in protocol['conditions']:
            b.reset();before=[probe(f) for f in frames]
            b.reset();b.weights_frozen=condition=='frozen';warm();start=time.perf_counter()
            cs_start=2200 if condition=='backward' else 0;cs_end=cs_start+1000
            pulses=protocol['backward_US_starts_ms'] if condition=='backward' else protocol['forward_US_starts_ms']
            if condition=='no_imposed_US':pulses=[]
            duration=max(cs_end,1900);train=np.zeros(b.n,dtype=np.int64);trace=[];delivered=0
            for ms in range(0,duration+2000,10):
                us=any(p<=ms<p+200 for p in pulses);delivered+=10*us
                c,t=span(frames[plus] if cs_start<=ms<cs_end else dark,10,
                    learning=condition!='frozen',stim=(b.circuit['dan'],us_current) if us else None)
                train+=c;trace+=t
            memory=b.memory();after=[probe(f) for f in frames]
            suppression=[1-sum(y['MBON'])/sum(x['MBON']) if sum(x['MBON']) else None for x,y in zip(before,after)]
            b.reset();erased=[probe(f) for f in frames]
            row={'plus':plus,'condition':condition,'before':before,'after':after,'erased':erased,'suppression':suppression,
                'selectivity':suppression[plus]-suppression[1-plus] if all(x is not None for x in suppression) else None,
                'memory':memory,'training_KC':int(train[b.circuit['kc']].sum()),'training_DAN':train[b.circuit['dan']].tolist(),
                'US_ms':delivered,'training_trace':trace,'wall_seconds':time.perf_counter()-start}
            save_json(out/f'{plus}-{condition}.json',row)
            slim={k:v for k,v in row.items() if k not in ['before','after','erased','training_trace']}
            slim['before_MBON']=[r['MBON'] for r in before];slim['after_MBON']=[r['MBON'] for r in after]
            slim['reset_restores']=all(a['MBON']==c['MBON'] and a['KC_pattern_sha256']==c['KC_pattern_sha256'] for a,c in zip(before,erased))
            rows.append(slim);print(slim,flush=True)
            save_json(out/'progress.json',{'completed':len(rows),'total':8,'latest':slim})
    passed=True
    for plus in [0,1]:
        r={x['condition']:x for x in rows if x['plus']==plus};p=r['paired']['selectivity']
        passed &= p is not None and p>0 and all(r[x]['selectivity'] is not None and p>r[x]['selectivity'] for x in ['backward','no_imposed_US'])
        passed &= r['frozen']['before_MBON']==r['frozen']['after_MBON'] and all(x['reset_restores'] for x in r.values())
    save_json(out/'results.json',{'complete':True,'development_gate':bool(passed),'independently_validated':False,'survival_learning_demonstrated':False,'rows':rows})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT/'physiology-v5/conditioning'));p.add_argument('--eta',type=float,default=.001);p.add_argument('--us-current',type=float,default=4.)
    a=p.parse_args();run(a.out,a.eta,a.us_current)
