"""Exploratory full-brain Doom survival runs, without a viewer or wall pacing.

This candidate has NOT passed physiological validation. Results cannot promote
themselves to an announcement claim. Every game tic renders a real RGB frame;
every neural step remains 0.1 ms, with fixed DNp20/DNpe017 button decoding.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from .calibration import calibrated_brain
from doom_learning.common import GRAPH,save_json,capture_provenance,digest,require_single_blas_thread
from doom_learning.controls import shifted_exposure
from doom_learning.survival_arena import SurvivalArena
from doom_learning_v2.vision import frame_for
from doom.engine import NeuralControls


def horizon_steps(seconds):return round(round(seconds*35)*10000/35)


def episode(b,seed,seconds,*,out,learning=False,freeze=True,schedule=None,vision=True,frames=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((GRAPH.parent/'manifest.json').read_text())
    controls=NeuralControls(manifest['readouts'],mode='bci')
    b.reset(keep_memory=True);b.weights_frozen=freeze
    wall=time.perf_counter();kernel=0
    # Same physiological equilibration in each independent episode.
    _,t=b.rgb_step(frame_for('black'),2000,learning=False);kernel+=t
    origin=b.cursor;until=-1;rows=[];horizon=round(seconds*35)
    delivered=np.zeros(horizon_steps(seconds),dtype=bool)
    if schedule is not None and (schedule.dtype!=bool or schedule.shape!=delivered.shape):raise ValueError('Wrong neural-step US schedule')
    a=SurvivalArena(out/'arena.wad',seed,seconds=seconds)
    previous=a.observation()['health'];initial_memory=b.memory()
    try:
        for tick in range(horizon):
            if a.observation()['finished']:break
            rgb=a.pixels();sensory=rgb if vision else np.zeros_like(rgb)
            # Round cumulative time, not each frame independently: 35 game
            # tics always correspond to exactly 10,000 neural integration steps.
            steps=round((tick+1)*10000/35)-(b.cursor-origin)
            begin=b.cursor-origin
            pulse=(schedule[begin:begin+steps] if schedule is not None else np.arange(b.cursor,b.cursor+steps)<until)
            delivered[begin:begin+steps]=pulse
            # A pulse can end between game tics. Split only at that boundary,
            # preserving exactly 2,000 neural steps for a 200 ms stimulus.
            boundaries=np.r_[0,np.flatnonzero(pulse[1:]!=pulse[:-1])+1,steps]
            c=np.zeros(b.n,dtype=np.int32)
            for lo,hi in zip(boundaries[:-1],boundaries[1:]):
                part,t=b.rgb_step(sensory,(hi-lo)*.1,learning=learning,
                    stimulation=(b.circuit['dan'],4.) if pulse[lo] else None)
                c+=part;kernel+=t
            action=controls.decode(c,steps*.0001)
            a.act(action);state=a.observation()
            damage=max(0,previous-state['health']);previous=state['health']
            if schedule is None and damage>0:until=b.cursor+2000
            # Observer-only coordinates do not enter rgb_step or decode.
            row={'tick':tick+1,'seconds':(tick+1)/35,'brain_steps':b.cursor-origin,
                 **state,'position':a.observer_position(),'damage':damage,'US_active':bool(pulse.any()),'US_steps':int(pulse.sum()),
                 'KC_spikes':int(c[b.circuit['kc']].sum()),'DAN_spikes':c[b.circuit['dan']].tolist(),
                 'MBON_spikes':c[b.circuit['mb']].tolist(),
                 'action':{k:action[k] for k in ['turn','forward','attack']},
                 'frame_sha256':digest(rgb),'sensory_sha256':digest(sensory),'spikes_sha256':digest(c),
                 'memory':b.memory()}
            rows.append(row)
            if tick%35==0:
                save_json(out/'current.json',{'status':'running','seed':seed,'state':row})
                if frames:
                    from PIL import Image
                    f=out/'frames';f.mkdir(exist_ok=True);Image.fromarray(rgb).save(f'{f}/{tick:05d}.png')
        elapsed=time.perf_counter()-wall;state=a.observation();sim=(b.cursor-origin)*.0001
        record={'seed':seed,'seconds_limit':seconds,'learning_enabled':learning,'weights_frozen':freeze,
            'vision':vision,'game_tics':len(rows),'brain_seconds':sim,'survival_seconds':len(rows)/35,
            'dead':state['dead'],'right_censored':not state['dead'],'end_health':state['health'],
            'damage_total':sum(r['damage'] for r in rows),'US_tics':sum(r['US_active'] for r in rows),
            'US_ms':float(delivered.sum()*.1),
            'before':initial_memory,'after':b.memory(),'assets':a.assets,
            'timing':{'wall_seconds':elapsed,'kernel_seconds':kernel,'brain_seconds_including_warmup':sim+2,'speed':(sim+2)/elapsed},'trace':rows}
        save_json(out/'episode.json',record);save_json(out/'current.json',{'status':'complete','summary':{k:v for k,v in record.items() if k not in ['trace','assets']}})
        return record,delivered
    finally:a.close()


def run(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Fresh output directory required')
    out.mkdir(parents=True);capture_provenance(out,additional=['doom_learning_v2','doom_learning_v6'])
    protocol={'model':'adaptive-centered-v6','status':'exploratory; physiological validation failed/pending',
        'training_seeds':args.seeds,'heldout_seeds':args.eval_seeds,'episode_seconds':args.seconds,
        'training_episodes':args.train_episodes,'eta':args.eta,'arms':['plastic','frozen','shuffled'],
        'reinforcement':'Observed damage schedules 200 ms +4 mV-equivalent PPL101 stimulation from the next tic. No game state reaches the fixed decoder.',
        'test':'Weights frozen and imposed reinforcement off. Same held-out maps in every arm. Memory erasure and black sensory input tested separately.',
        'shuffling':'Circularly shift donor exposure; record actual delivered duration and mark mismatches, including early recipient deaths.',
        'fast_mode':'No UI or pacing. Every 35 Hz game frame and every 0.1 ms neural step retained. Two seconds dark equilibration per episode.',
        'claim_gate':'Requires independently validated physiology/conditioning and held-out survival benefit across independent replicas, frozen and timing-shuffled controls, retention and erasure. This exploratory runner cannot certify the claim.'}
    save_json(out/'protocol.json',protocol);b=calibrated_brain(args.eta);rows=[];donors={}
    save_json(out/'circuit.json',b.circuit['report']);save_json(out/'visual.json',b.visual_report)
    def record(r,replicate,mode,phase):
        slim={k:v for k,v in r.items() if k not in ['trace','assets']};slim.update(replicate=replicate,condition=mode,phase=phase)
        rows.append(slim);save_json(out/'progress.json',{'completed_episodes':len(rows),'latest':slim});print(slim,flush=True)
    for seed in args.seeds:
        for mode in ['plastic','frozen','shuffled']:
            b.reset();branch=out/f'{seed}-{mode}'
            for n in range(args.train_episodes):
                schedule=None
                if mode=='shuffled':schedule,shift=shifted_exposure(*donors[(seed,n)][:2],seed+700000+n)
                r,s=episode(b,seed+n*1000,args.seconds,out=branch/f'train-{n}',learning=mode!='frozen',freeze=mode=='frozen',schedule=schedule,frames=args.frames)
                if mode=='plastic':donors[(seed,n)]=(s.copy(),round(r['brain_seconds']*10000),r['US_ms'])
                if mode=='shuffled':
                    r['dose_matched']=bool(s.sum()==donors[(seed,n)][0].sum() and abs(r['US_ms']-donors[(seed,n)][2])<=.11)
                    r['shift_neural_steps']=shift;save_json(branch/f'train-{n}/episode.json',r)
                record(r,seed,mode,'training')
            memory={k:getattr(b,k).copy() for k in ['memory_u','memory_w']}
            weights=b.weight[b.circuit['edges']].copy()
            np.savez(branch/'learned-efficacies.npz',weights=weights,**memory,
                metadata=json.dumps({'configuration':b.configuration_signature(),'note':'Memory-only artifact. Full-state checkpoint API is available separately.'}))
            def restore():
                b.reset()
                for k,v in memory.items():getattr(b,k)[:]=v
                b.weight[b.circuit['edges']]=weights
            for test_seed in args.eval_seeds:
                restore();r,_=episode(b,test_seed,args.seconds,out=branch/f'test-{test_seed}',schedule=np.zeros(horizon_steps(args.seconds),bool),frames=args.frames)
                record(r,seed,mode,'held_out')
            if mode=='plastic':
                restore();r,_=episode(b,args.eval_seeds[0],args.seconds,out=branch/'vision-black',vision=False,schedule=np.zeros(horizon_steps(args.seconds),bool))
                record(r,seed,mode,'vision_black')
                restore();b.weights_frozen=False;b.rgb_step(frame_for('black'),5000,learning=False)
                r,_=episode(b,args.eval_seeds[0],args.seconds,out=branch/'retention-5s',schedule=np.zeros(horizon_steps(args.seconds),bool))
                record(r,seed,mode,'retention_5s')
                b.reset();r,_=episode(b,args.eval_seeds[0],args.seconds,out=branch/'memory-erased',schedule=np.zeros(horizon_steps(args.seconds),bool))
                record(r,seed,mode,'memory_erased')
    result={'complete':True,'protocol':protocol,'episodes':rows,'survival_learning_demonstrated':False,'announcement_ready':False}
    save_json(out/'results.json',result)


if __name__=='__main__':
    require_single_blas_thread();p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/doom-learning/physiology-v6/survival-pilot')
    p.add_argument('--seeds',type=lambda x:list(map(int,x.split(','))),default=[41031,41032,41033])
    p.add_argument('--eval-seeds',type=lambda x:list(map(int,x.split(','))),default=[61031,61032,61033])
    p.add_argument('--seconds',type=float,default=30);p.add_argument('--train-episodes',type=int,default=2)
    p.add_argument('--eta',type=float,default=.001);p.add_argument('--frames',action='store_true');a=p.parse_args()
    train=[s+n*1000 for s in a.seeds for n in range(a.train_episodes)]
    if a.seconds<=0 or a.train_episodes<1 or not a.seeds or not a.eval_seeds:p.error('Positive durations/counts and nonempty seeds required')
    if len(set(train))!=len(train) or len(set(a.eval_seeds))!=len(a.eval_seeds) or set(train)&set(a.eval_seeds):p.error('Training and test seeds must be distinct and nonoverlapping')
    run(a)
