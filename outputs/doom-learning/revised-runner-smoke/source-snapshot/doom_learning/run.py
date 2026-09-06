"""Unpaced, full-connectome survival experiments with matched control arms.

Every game tic still receives 285/286 0.1-ms neural steps and a fresh rendered
RGB observation. Headless means no viewer/JPEG/history serialization or wall-
clock sleeping; it does not mean skipping sensory frames or simulation time.
"""
import argparse,json,time
from pathlib import Path
from .common import require_single_blas_thread, ROOT, GRAPH, OUT, save_json, digest, capture_provenance


def episode(brain, seed, seconds, *, learning=False, vision='intact', stimulation_schedule=None,
            record_frames=False, out=None):
    import numpy as np
    from doom.engine import NeuralControls
    from doom.game import Game,retinal_samples
    manifest=json.loads((GRAPH.parent/'manifest.json').read_text())
    controls=NeuralControls(manifest['readouts'],mode='bci')
    brain.reset(keep_memory=True)
    game=Game(seed=seed,scenario='health_gathering')
    game.game.set_episode_timeout(round(seconds*35))
    # ViZDoom permits timeout changes at runtime; a fresh episode ensures the
    # same episode/seed sequence in every arm, including the first episode.
    game.new_episode()
    rows=[];pulses=[];until_ms=-1.;start=time.perf_counter();kernel=0.;sensory=0.;game_wall=0.;previous_health=game.observation()['health'];initial_health=previous_health;initial_game_tic=game.game.get_episode_time()
    total_kc=0;total_dan=0;initial_memory=brain.memory();horizon=round(seconds*35)
    if stimulation_schedule is not None and len(stimulation_schedule)!=horizon:raise ValueError('Wrong reinforcement schedule length')
    try:
        for tick in range(horizon):
            if game.game.is_episode_finished():break
            t=time.perf_counter();frame=game.pixels();light=retinal_samples(frame,brain.uv)
            if vision=='black':light.fill(0)
            elif vision!='intact':raise ValueError('Unknown visual condition')
            sensory+=time.perf_counter()-t
            active=(bool(stimulation_schedule[tick]) if stimulation_schedule is not None else brain.sim_ms<until_ms)
            pulses.append(active)
            steps=round((tick+1)*10000/35)-brain.cursor
            counts,wall=brain.step(light,steps*.1,learning=learning,
                stimulation=(brain.circuit['dan'],30.) if active else None)
            kernel+=wall;action=controls.decode(counts,steps*.0001)
            t=time.perf_counter();game.act(action);state=game.observation();game_wall+=time.perf_counter()-t
            # Game health is used only by the explicitly engineered US adapter.
            # No health, reward, coordinates or object labels enter the decoder.
            damage=max(0,previous_health-state['health']);healing=max(0,state['health']-previous_health)
            previous_health=state['health']
            if stimulation_schedule is None and damage>0:until_ms=brain.sim_ms+200.
            kc=int(counts[brain.circuit['kc']].sum());dan=int(counts[brain.circuit['dan']].sum())
            total_kc+=kc;total_dan+=dan
            row={'tick':tick+1,'brain_seconds':brain.sim_ms/1000,'health':state['health'],'damage':damage,'healing':healing,
                'US_active':active,'KC_spikes':kc,'DAN_spikes':dan,'MBON_spikes':counts[brain.circuit['mb']].tolist(),
                'input_rgb_sha256':digest(frame),'retinal_sha256':digest(light),'spikes_sha256':digest(counts),
                'action':{k:action[k] for k in ['turn','forward','attack']},'memory':brain.memory()}
            rows.append(row)
            if out is not None and (tick%35==0 or state['finished']):
                save_json(Path(out)/'current.json',{'status':'running','seed':seed,'learning_enabled':learning,
                    'scenario':'health_gathering','vision':vision,'state':row,'wall_seconds':time.perf_counter()-start})
            if record_frames and out is not None and tick%35==0:
                from PIL import Image
                folder=Path(out)/'frames';folder.mkdir(parents=True,exist_ok=True)
                Image.fromarray(frame).save(folder/f'{tick:06d}.png')
        elapsed=time.perf_counter()-start;state=game.observation()
        result={'seed':seed,'scenario':'health_gathering','horizon_seconds':seconds,'game_tics':len(rows),
            'initial_health':initial_health,'initial_game_tic':initial_game_tic,'final_game_tic':game.game.get_episode_time(),
            'survival_seconds':len(rows)/35,'died':state['health']<=0,'right_censored':state['health']>0,
            'end_health':state['health'],'healing_total':sum(x['healing'] for x in rows),
            'damage_total':sum(x['damage'] for x in rows),'US_tics':sum(pulses),
            'KC_spikes':total_kc,'DAN_spikes':total_dan,'before':initial_memory,'after':brain.memory(),
            'timing':{'wall_seconds':elapsed,'neural_seconds':brain.sim_ms/1000,'speed':brain.sim_ms/1000/elapsed,
                'kernel_seconds':kernel,'sensory_seconds':sensory,'game_seconds':game_wall},
            'assets':game.assets,'trace':rows}
        schedule=np.zeros(horizon,dtype=bool);schedule[:len(pulses)]=pulses
        return result,schedule
    finally:game.close()


def run(args):
    import numpy as np
    from .brain import MemoryBrain,PARAMETERS,MODEL
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory; recorded experiments are never overwritten.')
    out.mkdir(parents=True,exist_ok=True);capture_provenance(out)
    b=MemoryBrain(eta=args.eta)
    protocol={'schema':2,'model':MODEL,'parameters':{**PARAMETERS,'eta_per_pair':args.eta},
        'train_seeds':args.seeds,'eval_seeds':args.eval_seeds,'train_episodes':args.train_episodes,
        'episode_seconds':args.episode_seconds,'retention_rest_seconds':args.retention_seconds,
        'conditions':['plastic','frozen','shuffled'],'scenario':'health_gathering',
        'controls':'Same fixed DNp20/DNpe017 BCI in every arm. No trained or game-state-driven decoder.',
        'reinforcement':'A health decrease schedules 200 ms of +30 mV-equivalent PPL101 current starting next game tic.',
        'shuffling':'Circular shift within the paired arm actual observed interval, then zero-pad to the planned horizon. Actual delivered exposure is still checked; earlier recipient deaths can prevent dose matching.',
        'headless':'No sleep, viewer, compressed video or high-rate full-state serialization. RGB generated and sampled every tic.',
        'experimental_unit':'Independent game seeds and reset copies of one reconstructed male; not independent biological flies.',
        'evaluation':'Plasticity and all imposed reinforcement off; matched held-out seeds. Neural dynamics reset between episodes; learned efficacies retained.',
        'retention':'Quiet zero-drive simulated interval followed by reset of fast neural state. LTD-only persistence is a model property, not measured biological retention.',
        'claim_gate':'Vision pathway, conditioning, dose-matched comparisons, held-out improvement and reset control must all pass. A working runner is not proof of learning.'}
    save_json(out/'protocol.json',protocol);save_json(out/'circuit.json',b.circuit['report'])
    all_rows=[];yokes={}
    for seed in args.seeds:
        for mode in ['plastic','frozen','shuffled']:
            b.reset();branch=out/f'{seed}-{mode}';branch.mkdir(exist_ok=True)
            for n in range(args.train_episodes):
                train_seed=seed+n*1000
                yoke=None
                if mode=='shuffled':
                    from .controls import shifted_exposure
                    original,observed=yokes[(seed,n)]
                    yoke,shift=shifted_exposure(original,observed,seed+n+700000)
                result,schedule=episode(b,train_seed,args.episode_seconds,learning=mode!='frozen',
                    stimulation_schedule=yoke,out=branch/f'training-{n+1}',record_frames=args.frames)
                if mode=='plastic':yokes[(seed,n)]=(schedule.copy(),result['game_tics'])
                if mode=='shuffled':
                    result['US_dose_matched']=int(schedule.sum())==int(yokes[(seed,n)][0].sum())
                    result['US_shift_tics']=shift
                result.update({'replicate':seed,'condition':mode,'phase':'training','training_episode':n+1})
                all_rows.append(result);save_json(branch/f'train-{n+1}.json',result)
                print(json.dumps({k:v for k,v in result.items() if k not in ['trace','assets']}),flush=True)
            b.checkpoint(branch/'trained.npz')
            # Every test uses a fresh fast state with exactly the trained weights.
            for eval_seed in args.eval_seeds:
                result,_=episode(b,eval_seed,args.episode_seconds,learning=False,
                    stimulation_schedule=np.zeros(round(args.episode_seconds*35),dtype=bool))
                result.update({'replicate':seed,'condition':mode,'phase':'held_out','training_episode':args.train_episodes})
                all_rows.append(result);save_json(branch/f'eval-{eval_seed}.json',result)
            if mode=='plastic':
                b.reset(keep_memory=True)
                if args.retention_seconds:
                    b.step(np.zeros(len(b.retina)),args.retention_seconds*1000,learning=False,lamina_bias=0)
                result,_=episode(b,args.eval_seeds[0],args.episode_seconds,learning=False,
                    stimulation_schedule=np.zeros(round(args.episode_seconds*35),dtype=bool))
                result.update({'replicate':seed,'condition':mode,'phase':'retention','training_episode':args.train_episodes})
                all_rows.append(result);save_json(branch/'retention.json',result)
                b.reset()
                result,_=episode(b,args.eval_seeds[0],args.episode_seconds,learning=False,
                    stimulation_schedule=np.zeros(round(args.episode_seconds*35),dtype=bool))
                result.update({'replicate':seed,'condition':mode,'phase':'reset_memory','training_episode':args.train_episodes})
                all_rows.append(result);save_json(branch/'reset-memory.json',result)
            save_json(out/'progress.json',{'status':'running','finished_branch':str(branch.name),'completed_episodes':len(all_rows)})
    # Preserve all per-tic data in branch records and a compact comparison file.
    report={'schema':1,'status':'complete','protocol':protocol,'circuit':b.circuit['report'],
            'episodes':[{k:v for k,v in r.items() if k not in ['trace','assets']} for r in all_rows],
            'learning_demonstrated':False,'biologically_validated':False}
    save_json(out/'results.json',report)
    save_json(out/'progress.json',{'status':'complete','completed_episodes':len(all_rows)})
    return report


if __name__=='__main__':
    require_single_blas_thread();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(OUT/'survival'))
    p.add_argument('--seeds',type=lambda s:[int(x) for x in s.split(',')],default=[41031,41032,41033])
    p.add_argument('--eval-seeds',type=lambda s:[int(x) for x in s.split(',')],default=[61031,61032,61033])
    p.add_argument('--train-episodes',type=int,default=2);p.add_argument('--episode-seconds',type=float,default=30.)
    p.add_argument('--retention-seconds',type=float,default=5.);p.add_argument('--eta',type=float,default=.001)
    p.add_argument('--frames',action='store_true');args=p.parse_args()
    if args.train_episodes<1 or args.episode_seconds<=0 or args.retention_seconds<0 or not args.seeds or not args.eval_seeds:p.error('Positive episode duration/count and nonempty seed lists required')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.eval_seeds))!=len(args.eval_seeds):p.error('Seeds must be unique')
    if len({s+i*1000 for s in args.seeds for i in range(args.train_episodes)})!=len(args.seeds)*args.train_episodes:p.error('Derived training seeds overlap between replicas')
    if set(args.eval_seeds)&{s+i*1000 for s in args.seeds for i in range(args.train_episodes)}:p.error('Evaluation seeds must be held out')
    run(args)
