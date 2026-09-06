"""Conservative evidence gates and viewer-sized, directly traceable results."""
import argparse,json,statistics
from datetime import datetime,timezone
from pathlib import Path
from .common import OUT,save_json


def load(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def summarize(survival=OUT/'survival-pilot',out=OUT/'report.json'):
    survival=Path(survival)
    vision=load(OUT/'vision.json',{});conditioning=load(OUT/'conditioning.json',{})
    result=load(survival/'results.json',{});rows=result.get('episodes',[])
    modes=['plastic','frozen','shuffled'];comparison=[]
    for mode in modes:
        tests=[r for r in rows if r['condition']==mode and r['phase']=='held_out']
        train=[r for r in rows if r['condition']==mode and r['phase']=='training']
        comparison.append({'condition':mode,'episodes':len(tests),'training_replicates':len(set(r['replicate'] for r in tests)),
            'mean_survival_seconds':statistics.mean(r['survival_seconds'] for r in tests) if tests else None,
            'deaths':sum(r['died'] for r in tests),'censored':sum(r['right_censored'] for r in tests),
            'changed_edges_max':max((r['after']['changed_edges'] for r in train),default=0),
            'training_KC_spikes':sum(r['KC_spikes'] for r in train),
            'training_DAN_spikes':sum(r['DAN_spikes'] for r in train),
            'delivered_US_tics':sum(r['US_tics'] for r in train)})
    pairwise=[]
    for other in ['frozen','shuffled']:
        for rep in sorted({r['replicate'] for r in rows}):
            pairs=[]
            for r in rows:
                if r['replicate']!=rep or r['condition']!='plastic' or r['phase']!='held_out':continue
                match=next((s for s in rows if s['replicate']==rep and s['seed']==r['seed'] and s['condition']==other and s['phase']=='held_out'),None)
                if match:pairs.append(r['survival_seconds']-match['survival_seconds'])
            if pairs:pairwise.append({'replicate':rep,'versus':other,'mean_paired_difference_seconds':statistics.mean(pairs)})
    dose_rows=[r for r in rows if r['condition']=='shuffled' and r['phase']=='training']
    dose_matched=bool(dose_rows) and all(r.get('US_dose_matched',False) for r in dose_rows)
    memory_changed=any(r['condition']=='plastic' and r['phase']=='training' and r['after']['changed_edges']>0 for r in rows)
    vision_active=bool(vision.get('visual_memory_input_active')) and any(p.get('different_visual_KCs',0)>0 for p in vision.get('pairwise',[]))
    conditioning_passed=bool(conditioning.get('qualitative_selectivity_and_timing_passed'))
    status='complete' if result.get('status')=='complete' and conditioning else 'running'
    patterns=[{'stimulus':r['stimulus'],'receptor_spikes':r['groups']['receptors']['spikes'],
        'T4_spikes':r['groups']['T4']['spikes'],'T5_spikes':r['groups']['T5']['spikes'],
        'KC_spikes':r['groups']['Kenyon']['spikes'],'MBON_spikes':r['groups']['MBON11']['spikes']}
        for r in vision.get('conditions',[])]
    conditioning_rows=[]
    for r in conditioning.get('conditions',[]):
        name='no_imposed_US' if r['condition']=='no_dopamine' else r['condition']
        conditioning_rows.append({'condition':name,'changed_edges':r['memory_after_training']['changed_edges'],
            'mean_efficacy':r['memory_after_training']['mean_efficacy'],
            'before_MBON_spikes':[sum(x['MBON_spikes']) for x in r['before']],
            'after_MBON_spikes':[sum(x['MBON_spikes']) for x in r['after']],
            'DAN_spikes':r['training_DAN_spikes']})
    traces=[]
    if rows:
        rep=min(r['replicate'] for r in rows);eval_seed=min(r['seed'] for r in rows if r['phase']=='held_out')
        for mode in modes:
            record=load(survival/f'{rep}-{mode}'/f'eval-{eval_seed}.json')
            if record:
                raw=record['trace'];points=raw[::7]
                if raw and (not points or points[-1]['tick']!=raw[-1]['tick']):points.append(raw[-1])
                traces.append({'condition':mode,'replicate':rep,'seed':eval_seed,'survival_seconds':record['survival_seconds'],
                    'points':[{'seconds':p['brain_seconds'],'health':p['health'],'efficacy':p['memory']['mean_efficacy']} for p in points]})
    timings=[r['timing'] for r in rows]
    report={'schema':1,'generated_at':datetime.now(timezone.utc).isoformat(),'status':status,
        'title':'Learning laboratory','verdict':'Learning not demonstrated','learning_demonstrated':False,
        'live_baseline_changed':False,'candidate_publicly_promoted':False,
        'gates':[{'name':'Visual memory input','passed':vision_active,'detail':'Controlled images must activate and distinguish memory-cell responses.'},
                 {'name':'Selective conditioning','passed':conditioning_passed,'detail':'Paired stimulation must affect its cue more than control cues and backward/no-US conditions.'},
                 {'name':'Survival memory changes','passed':memory_changed,'detail':'Game experience must change the identified memory synapses.'},
                 {'name':'Matched reinforcement exposure','passed':dose_matched,'detail':'Shuffled controls must receive the same amount of imposed stimulation.'}],
        'vision':patterns,'conditioning':conditioning_rows,'comparison':comparison,'paired_survival_differences':pairwise,
        'traces':traces,'protocol':result.get('protocol',load(survival/'protocol.json',{})),
        'retention_and_reset':[r for r in rows if r['phase'] in ['retention','reset_memory']],
        'performance':{'episodes':len(timings),'wall_seconds':sum(t['wall_seconds'] for t in timings),
            'brain_seconds':sum(t['neural_seconds'] for t in timings),
            'aggregate_speed':sum(t['neural_seconds'] for t in timings)/sum(t['wall_seconds'] for t in timings) if timings else None},
        'limits':['One reconstructed male, repeated across game seeds; no independent biological animals.',
            'This is a small development pilot, not a confirmatory survival study.',
            'The imposed-KC assay bypasses vision and does not reproduce the original odor experiment.',
            'No imposed US does not silence endogenous dopamine neurons.',
            'Quiet-delay retention in an LTD-only model is a modeled property, not proof of biological retention.',
            'Failed gates prevent a learning claim or promotion to the live broadcast.'],
        'sources':[{'title':'Presynaptic dopamine-dependent depression','url':'https://doi.org/10.1016/j.neuron.2015.11.003'},
                   {'title':'Compartment-specific memory rules','url':'https://doi.org/10.7554/eLife.16135'},
                   {'title':'Visual and olfactory memory circuits','url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC4135349/'},
                   {'title':'Visual-system modeling and physiological constraints','url':'https://www.nature.com/articles/s41586-024-07939-3'},
                   {'title':'ViZDoom survival scenario','url':'https://vizdoom.farama.org/environments/default/'}]}
    save_json(out,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--survival',default=str(OUT/'survival-pilot'))
    p.add_argument('--out',default=str(OUT/'report.json'));a=p.parse_args()
    r=summarize(a.survival,a.out);print(json.dumps({'status':r['status'],'gates':r['gates'],'performance':r['performance']}))
