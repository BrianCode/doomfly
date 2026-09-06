"""Full-graph imposed-activity conditioning assay, separate from visual learning.

Tests a qualitative Hige-inspired timing/selectivity prediction. Artificial
current stimulation of identified KCs bypasses vision; this cannot pass the
visual-conditioning gate or establish replication of the original odor assay.
"""
import argparse,json,time
from .common import require_single_blas_thread, OUT, save_json, annotations


def run(out=OUT/'conditioning.json'):
    import numpy as np
    from .brain import MemoryBrain,PARAMETERS
    b=MemoryBrain();a=annotations(b.ids)
    candidates=np.flatnonzero(a.type.eq('KCg-d'))
    # Deterministic, disjoint imposed cell ensembles, not odor- or pixel-selective
    # ensembles inferred from the animal. No game task is involved here.
    cues=[candidates[::2][:64],candidates[1::2][:64]]
    dark=np.zeros(len(b.retina));mb=b.circuit['mb'];dan=b.circuit['dan']
    def probe(cue):
        b.reset(keep_memory=True)
        c,_=b.step(dark,1000,learning=False,stimulation=(cue,30.),lamina_bias=0)
        return {'MBON_spikes':c[mb].tolist(),'KC_spikes':int(c[cue].sum()),'DAN_spikes':int(c[dan].sum())}
    rows=[]
    for condition in ['paired','backward','frozen','no_imposed_US','reset_memory']:
        b.reset();before=[probe(c) for c in cues];b.reset();start=time.perf_counter()
        cs_start=2000 if condition=='backward' else 0
        cs_end=cs_start+1000
        pulse_starts=[0,500,1000,1500] if condition=='backward' else [200,700,1200,1700]
        if condition=='no_imposed_US':pulse_starts=[]
        duration=max(cs_end,1800)
        bounds=sorted(set([0,cs_start,cs_end,duration,*pulse_starts,*[t+1 for t in pulse_starts]]))
        train_kc=0;train_dan=0
        for begin,end in zip(bounds,bounds[1:]):
            stimulation=[]
            if cs_start<=begin<cs_end:stimulation.append((cues[0],30.))
            if begin in pulse_starts:stimulation.append((dan,200.))
            c,_=b.step(dark,end-begin,learning=condition!='frozen',stimulation=stimulation,lamina_bias=0)
            train_kc+=int(c[b.circuit['kc']].sum());train_dan+=int(c[dan].sum())
        memory=b.memory()
        if condition=='reset_memory':b.reset()
        after=[probe(c) for c in cues]
        row={'condition':condition,'before':before,'after':after,'memory_after_training':memory,
             'memory_at_test':b.memory(),'training_KC_spikes':train_kc,'training_DAN_spikes':train_dan,
             'wall_seconds':time.perf_counter()-start}
        rows.append(row);print(json.dumps(row),flush=True)
    paired=rows[0];suppression=[]
    for before,after in zip(paired['before'],paired['after']):
        n=sum(before['MBON_spikes']);suppression.append(1-sum(after['MBON_spikes'])/n if n else None)
    qualitative=(all(x is not None for x in suppression) and suppression[0]>suppression[1] and
        paired['memory_after_training']['changed_edges']>0 and
        all(r['memory_after_training']['changed_edges']==0 for r in rows if r['condition'] in ['backward','frozen','no_imposed_US']) and
        rows[-1]['before']==rows[-1]['after'])
    report={'schema':1,'assay':'imposed-KC-activity, full graph','circuit':b.circuit['report'],'parameters':PARAMETERS,
        'conditions':rows,'paired_MBON_suppression_CS_plus_CS_minus':suppression,
        'qualitative_selectivity_and_timing_passed':bool(qualitative),
        'visual_conditioning_demonstrated':False,'original_experiment_reproduced':False,
        'stimulus':{'KC_current_mv_equivalent':30,'KC_cells_per_cue':64,'DAN_current_mv_equivalent':200,
                    'DAN_pulse_ms':1,'DAN_pulses':4,'different_sensory_modality_from_paper':True},
        'limits':'Technical and qualitative mechanistic check with imposed KC activity. The original work used odor-driven populations, living physiology and longer inter-trial intervals. Output suppression alone is not behavior or a quantitative experimental replication.',
        'evidence':['https://doi.org/10.1016/j.neuron.2015.11.003','https://doi.org/10.7554/eLife.16135']}
    save_json(out,report);return report


if __name__=='__main__':
    require_single_blas_thread();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(OUT/'conditioning.json'))
    run(p.parse_args().out)
