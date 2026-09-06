"""Publish measured iteration records, preserving the original failed pilot."""
import hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
from .common import ROOT,OUT,save_json


def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    conditioning=read(OUT/'physiology-v5/conditioning/results.json')
    stability=read(OUT/'physiology-v6/stability/results.json')
    survival=read(OUT/'physiology-v6/survival-pilot/results.json')
    arena=read(OUT/'survival-arena/check/environment-check.json')
    if not all([conditioning,stability,survival,arena]):raise RuntimeError('Wait for every recorded experiment to complete before publishing.')
    if not conditioning['complete'] or not stability['complete'] or not survival['complete']:raise RuntimeError('Incomplete experiment')
    rows=survival['episodes'];groups=[]
    for mode in ['plastic','frozen','shuffled']:
        tests=[r for r in rows if r['condition']==mode and r['phase']=='held_out']
        train=[r for r in rows if r['condition']==mode and r['phase']=='training']
        groups.append({'condition':mode,'test_episodes':len(tests),'training_replicates':len({r['replicate'] for r in train}),
            'mean_observed_survival':sum(r['survival_seconds'] for r in tests)/len(tests),
            'deaths':sum(r['dead'] for r in tests),'censored':sum(r['right_censored'] for r in tests),
            'changed_edges':max(r['after']['changed_edges'] for r in train),
            'test_results':[{'seed':r['seed'],'seconds':r['survival_seconds'],'censored':r['right_censored']} for r in tests]})
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'announcement_ready':False,
        'claim':'Survival learning has not been demonstrated. Current physiology and conditioning do not pass validation.',
        'dataset':{'name':'MaleCNS v1.0','neurons':166700,'directed_edges':25582938,'synaptic_contacts':124177617},
        'candidates':6,
        'vision':{'candidate':'v2','black_gamma_d_spikes':0,'blue_gamma_d_spikes':283,'blue_gamma_d_cells_active':14,'gamma_d_cells':206,
                  'interpretation':'A one-second cold-start signal-propagation assay. Distinct from warmed, recurrent-network validation. No demonstrated useful visual behavior.'},
        'conditioning':{'candidate':'v5','development_gate':conditioning['development_gate'],'rows':conditioning['rows']},
        'stability':{'candidate':'v6','rows':stability['rows']},
        'survival':{'candidate':'v6','complete':True,'groups':groups,'episodes':rows,
            'horizon_seconds':survival['protocol']['episode_seconds'],'independent_validation':False,
            'note':'Exploratory technical pilot with one training seed and two held-out starts. A capped episode is right-censored. Internal weight changes do not prove learning.'},
        'arena':{'engine_check_passed':arena['environment_test_passed'],'fly_behavior_in_engine_check':False,
                 'stationary_seconds':arena['rows'][0]['samples'][-1]['tick']/35,
                 'cross_to_safe_observed_seconds':arena['rows'][1]['samples'][-1]['tick']/35},
        'performance':{'timing_scope':'Recorded episode timers including warmup; excluding model construction and the separate five-second retention interval.','episodes':len(rows),'brain_seconds':sum(r['timing']['brain_seconds_including_warmup'] for r in rows),
            'wall_seconds':sum(r['timing']['wall_seconds'] for r in rows),'kernel_seconds':sum(r['timing']['kernel_seconds'] for r in rows)},
        'limits':['A reconstructed male wiring diagram with chosen dynamics, not a living or literal fly brain.',
            'R8 input projection and display spectra are inferred; photoreceptors and APL still lack calibrated graded physiology.',
            'The centered plasticity rule is adapted from a reduced published model; full-graph parameter transfer is unvalidated.',
            'Cue-induced persistent activity and no-punishment effects invalidate the current conditioning interpretation.',
            'No claim of survival learning, consciousness, natural fly behavior, or priority over prior work is supported.'],
        'sources':[{'title':'R8 cotransmission · Xiao et al. 2023','url':'https://doi.org/10.1038/s41586-023-06681-6'},
                   {'title':'Visual inputs to fly memory · 2024','url':'https://doi.org/10.1038/s41467-024-49616-z'},
                   {'title':'Dopamine-gated plasticity · Hige et al. 2015','url':'https://doi.org/10.1016/j.neuron.2015.11.003'},
                   {'title':'Baseline activity and memory dynamics · Huang, Luo et al. 2024','url':'https://doi.org/10.1038/s41586-024-07819-w'}]}
    save_json(ROOT/'doom-ui/data/learning-iterations.json',report)
    save_json(ROOT/'doom-ui/public/learning-iterations.json',report)
    # Explicit allowlist: no environment files, credentials, full graph arrays,
    # hidden directories, compiled binaries or arbitrary workspace contents.
    sources=[]
    for folder in ['doom','doom_learning',*[f'doom_learning_v{i}' for i in range(2,7)],'tests']:
        sources += [p for p in (ROOT/folder).iterdir() if p.suffix in ['.py','.cpp','.md'] and (folder!='tests' or p.name.startswith('test_doom') or p.name=='test_connectome.py')]
    sources += [ROOT/'docs/doom-learning-iteration-log.md',ROOT/'research/huang-2024/targets.json']
    for folder in [*[f'physiology-v{i}' for i in range(2,7)],'survival-arena']:
        sources += [p for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix in ['.json','.py','.cpp','.wad']]
    # External workbooks are retrieved from the cited publisher, never bundled.
    sources += [ROOT/'LICENSE', ROOT/'THIRD_PARTY.md', ROOT/'THIRD_PARTY_NOTICES.md',
                ROOT/'research/huang-2024/README.md', ROOT/'doom/datasets.json']
    sources += [p for p in (ROOT/'licenses').rglob('*') if p.is_file()]
    archive=ROOT/'doom-ui/public/learning-iterations-source.zip';manifest={}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(sources)):
            data=p.read_bytes();name=str(p.relative_to(ROOT));manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('bundle-manifest.json',json.dumps(manifest,indent=2)+'\n')
    save_json(ROOT/'doom-ui/public/learning-iterations-source-manifest.json',{'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':manifest})
    print(json.dumps({'complete':True,'announcement_ready':False,'episodes':len(rows),'archive_bytes':archive.stat().st_size,'files':len(manifest)}))


if __name__=='__main__':main()
