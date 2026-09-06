"""Mechanical integrity checks and honest, descriptive pilot comparisons."""
import argparse,json
from pathlib import Path
from doom_learning.common import save_json


def analyze(folder):
    p=Path(folder);r=json.loads((p/'results.json').read_text());episodes=r['episodes']
    records=[json.loads(f.read_text()) for f in sorted(p.glob('*/*/episode.json'))]
    checks={
        'complete_episode_count':len(records)==len(episodes),
        'neural_clock_within_half_step':all(abs(x['brain_steps']*.0001-x['seconds'])<=.000050001 for e in records for x in e['trace']),
        'one_observation_per_game_tic':all(len(e['trace'])==e['game_tics'] and [x['tick'] for x in e['trace']]==list(range(1,e['game_tics']+1)) for e in records),
        'frozen_efficacies_unchanged':all(e['before']['sha256']==e['after']['sha256'] for e in records if e['weights_frozen']),
        'tests_have_no_imposed_reinforcement':all(e['US_ms']==0 for e in episodes if e['phase']!='training'),
        'dose_matches_exact_neural_steps':all(abs(e['US_ms']-.1*sum(x['US_steps'] for x in e['trace']))<1e-8 for e in records),
        'shuffled_actual_dose_matched':all(e.get('dose_matched') is True for e in episodes if e['phase']=='training' and e['condition']=='shuffled'),
        'recorded_input_and_spike_hashes':all(len(x[k])==64 for e in records for x in e['trace'] for k in ['frame_sha256','sensory_sha256','spikes_sha256']),
    }
    identity=[]
    for seed in r['protocol']['training_seeds']:
        for test_seed in r['protocol']['heldout_seeds'][:1]:
            erased=json.loads((p/f'{seed}-plastic/memory-erased/episode.json').read_text())
            frozen=json.loads((p/f'{seed}-frozen/test-{test_seed}/episode.json').read_text())
            keys=['frame_sha256','spikes_sha256','action','health','position','brain_steps']
            exact=len(erased['trace'])==len(frozen['trace']) and all(all(a[k]==b[k] for k in keys) for a,b in zip(erased['trace'],frozen['trace']))
            identity.append({'replicate':seed,'test_seed':test_seed,'trace_matches_frozen':exact})
    checks['memory_erasure_restores_full_frozen_trace']=all(x['trace_matches_frozen'] for x in identity)
    controls=[]
    for seed in r['protocol']['training_seeds']:
        for test_seed in r['protocol']['heldout_seeds']:
            row={'replicate':seed,'test_seed':test_seed}
            for mode in ['plastic','frozen','shuffled']:
                e=next(e for e in episodes if e['replicate']==seed and e['seed']==test_seed and e['phase']=='held_out' and e['condition']==mode)
                row[mode]={'seconds':e['survival_seconds'],'censored':e['right_censored']}
            row['observed_plastic_minus_frozen']=row['plastic']['seconds']-row['frozen']['seconds'];controls.append(row)
    report={'technical_checks':checks,'all_technical_checks_pass':all(checks.values()),'erasure_identity':identity,'held_out':controls,
        'announcement_ready':False,'survival_learning_demonstrated':False,
        'reason':'Failed upstream physiological/conditioning validation. One training replica, descriptive pilot only; observed/censored durations do not justify a general learning claim.',
        'timing_scope':'Episode timing includes each two-second warmup; excludes model construction and standalone five-second retention interval.'}
    save_json(p/'analysis.json',report);print(json.dumps(report,indent=2))
    if not all(checks.values()):raise AssertionError('Pilot technical integrity check failed; inspect preserved report')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='?',default='outputs/doom-learning/physiology-v6/survival-pilot');a=p.parse_args();analyze(a.folder)
