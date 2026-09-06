"""Shared, read-only live broadcast. The local process owns the neural/game loop.

Only GET /state and /health are exposed. No filesystem, shell, credentials,
remote controls, or model-mutating endpoint. Stale frames never become replays.
"""
import argparse,base64,hashlib,io,json,threading,time,uuid,logging
from logging.handlers import RotatingFileHandler
from collections import deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
from doom.native import NativeBrain,BUILD
from doom.engine import NeuralControls
from doom.game import Game,retinal_samples
from doom.reward import SugarReinforcement
from doom.provenance import provenance
ROOT=Path(__file__).resolve().parents[1]
latest={'status':'starting','generated_at_ms':0}; stop=threading.Event()

def encoded_frame(rgb):
    f=io.BytesIO();Image.fromarray(rgb).save(f,format='JPEG',quality=75)
    return 'data:image/jpeg;base64,'+base64.b64encode(f.getvalue()).decode()

def run_loop(args):
    global latest
    try:
        manifest=json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
        brain=NativeBrain(ROOT/'outputs/doom/malecns_v1/graph.npz')
        controls=NeuralControls(manifest['readouts'],mode=args.decoder);game=Game(seed=args.seed)
        origin=provenance(ROOT/'outputs/doom/malecns_v1/graph.npz',BUILD,game.assets)
        reward=SugarReinforcement(args.reward=='sugar')
        if args.condition=='retina_disconnected':
            for i in brain.retina:brain.weight[brain.ptr[i]:brain.ptr[i+1]]=0
        if args.condition=='all_edges_disconnected':brain.weight.fill(0)
        run_id=str(uuid.uuid4());start=time.monotonic();tick=0;seq=0;last_publish=0
        total_actions=0;history=deque(maxlen=160);timeline=deque(maxlen=50);episodes=deque(maxlen=12)
        group_names=np.unique(brain.superclass);groups=[np.flatnonzero(brain.superclass==k) for k in group_names]
        # Fixed display sample: ascending graph indices in visual system, central
        # brain and descending population. Raster points are recorded spikes.
        import pyarrow.feather as feather
        nodes=feather.read_table(ROOT/'connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
        annotations=feather.read_table(ROOT/'connectome_data/malecns_v1/annotations.feather').to_pandas().set_index('bodyId')
        retina_sides=annotations.loc[brain.ids[brain.retina],'rootSide'].to_numpy()
        display=[]
        for superclass in ['ol_intrinsic','visual_projection','cb_intrinsic','descending_neuron']:
            inds=np.flatnonzero(nodes.superclass.eq(superclass).to_numpy())
            display.extend(inds[np.linspace(0,len(inds)-1,min(32,len(inds)),dtype=int)].tolist())
        display=np.asarray(display);window_counts=np.zeros(brain.n,dtype=np.int32);window_ms=0
        audit_path=ROOT/'outputs/doom'/'audit.jsonl'
        audit_handler=RotatingFileHandler(audit_path,maxBytes=20_000_000,backupCount=4)
        audit_logger=logging.getLogger('doom-audit');audit_logger.handlers=[audit_handler];audit_logger.setLevel(logging.INFO);audit_logger.propagate=False
        frozen=None
        print(json.dumps({'status':'running','run_id':run_id,'condition':args.condition,'port':args.port}),flush=True)
        try:
          while not stop.is_set():
            began=time.monotonic()
            before=game.observation()
            if before['finished']:
                episodes.append(before);game.new_episode()
            frame=game.pixels();light=retinal_samples(frame,brain.uv)
            if args.condition=='blank_vision':light.fill(0)
            if args.condition=='frozen_vision':
                if frozen is None:frozen=light.copy()
                light=frozen
            # Alternate 285/286 substeps to keep neural and Doom clocks aligned
            # to <0.1 ms. Never silently skip steps when the host is slow.
            tick+=1;target=int(round(tick*10000/35));steps=target-brain.cursor
            sugar_applied=reward.active(brain.sim_ms)
            counts,neural_wall=brain.step(light,steps*.1,sugar=sugar_applied)
            action=controls.decode(counts,steps*.1/1000)
            if args.condition=='controls_clamped':
                applied={**action,'turn':0.,'forward':0.,'attack':False}
            else:applied=action
            score=game.act(applied);reward.observe(score,brain.sim_ms)
            after=game.observation()
            nonzero=abs(applied['turn'])>1e-9 or abs(applied['forward'])>1e-9 or applied['attack']
            total_actions+=int(nonzero)
            event={'run_id':run_id,'decoder':args.decoder,'condition':args.condition,'reward_mode':args.reward,'tick':tick,'neural_ms':round(brain.sim_ms,3),'episode':game.episode,
              'input_sha256':hashlib.sha256(light.tobytes()).hexdigest(),
              'source_frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),
              'spike_counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
              'requested':{k:action[k] for k in ['turn','forward','attack']},
              'applied':{k:applied[k] for k in ['turn','forward','attack']},
              'readouts':action['readouts'],'reward':score,'sugar_applied':sugar_applied,
              'sugar_scheduled_for_next_step':reward.active(brain.sim_ms),'model_revision':BUILD['model_revision'],
              'input_episode':game.episode,'input_game_tick':game.tick-1,'output_game_tick':game.tick,
              'neural_interval_ms':round(steps*.1,3)}
            audit_logger.info(json.dumps(event,separators=(',',':')))
            timeline.append({'tick':tick,'spikes':int(counts.sum()),'action':event['applied']})
            window_counts+=counts;window_ms+=steps*.1
            now=time.monotonic()
            if now-last_publish>=.10:
                seq+=1;age=now-start
                history.append({'neural_ms':round(brain.sim_ms,1),'window_ms':round(window_ms,3),'counts':window_counts[display].tolist(),'population_spikes':int(window_counts.sum())})
                # Show the frame that actually supplied this action's input,
                # avoiding an image/telemetry mismatch or invented interpolation.
                jpeg=encoded_frame(frame)
                latest={'schema':1,'status':'running','run_id':run_id,'sequence':seq,
                  'generated_at_ms':int(time.time()*1000),'condition':args.condition,'decoder':args.decoder,
                  'frame':jpeg,'input_frame_sha256':event['source_frame_sha256'],
                  'display_jpeg_sha256':hashlib.sha256(base64.b64decode(jpeg.split(',',1)[1])).hexdigest(),
                  'provenance':origin,
                  'manifest':manifest,'clocks':{'wall_seconds':round(age,3),'neural_seconds':round(brain.sim_ms/1000,4),
                    'game_seconds':round(tick/35,4),'speed':round((brain.sim_ms/1000)/age,3),'brain_step_ms':round(neural_wall*1000,3)},
                  'game':after,'episodes':list(episodes),'action':event['applied'],'readouts':action['readouts'],
                  'total_spikes':brain.total_spikes,'window_spikes':int(window_counts.sum()),'window_ms':round(window_ms,2),
                  'total_action_ticks':total_actions,'neuron_voltage_mv':{r['id']:round(float(brain.v[r['index']]),3) for r in manifest['readouts']},
                  'populations':[{'name':str(k),'neurons':len(ix),'spikes':int(window_counts[ix].sum()),
                    'mean_rate_hz':float(window_counts[ix].sum()/len(ix)/(window_ms/1000))} for k,ix in zip(group_names,groups)],
                  'retina':{'uv':brain.uv[::8].round(4).tolist(),'luminance':light[::8].round(4).tolist(),
                    'neuron_ids':[str(x) for x in brain.ids[brain.retina[::8]]], 'side':retina_sides[::8].tolist(),
                    'filtered_luminance':brain.luminance[::8].round(4).tolist(),
                    'drive_mv':brain.drive[brain.retina[::8]].round(4).tolist(),
                    'spikes_last_step':counts[brain.retina[::8]].tolist(),
                    'measurement':'Raw normalized frame luminance; not firing rate or a reconstructed subjective view.',
                    'full_sample_count':len(light),'display_stride':8},
                  'raster':{'neuron_ids':[str(brain.ids[i]) for i in display],'bins':list(history)},
                  'timeline':list(timeline),'audit':event,
                  'reward':{'mode':args.reward,'sugar_pulses':reward.pulses,'active':reward.active(brain.sim_ms),'plasticity':False},
                  'protocol':{'dt_ms':brain.dt,'lamina_bias_mv':12,'retinal_gain_mv':30,'photoreceptor_half_saturation':.02,'seed':args.seed,'replicate':'one reconstructed male',
                    'input_filter':'10 ms discrete low-pass, updated once per game interval; resulting current held for that interval',
                    'frame_timing':'JPEG of pre-action input frame; game counters are post-action; raw RGB hash differs from compressed JPEG hash',
                    'automatic_episode_reset':True,'neural_state_persists_across_episodes':True,'hosting':'local broadcaster; unavailable if host sleeps or disconnects'}}
                last_publish=now;window_counts.fill(0);window_ms=0
            remaining=start+tick/35-time.monotonic()
            if remaining>0:stop.wait(remaining)
        finally:
            game.close();audit_handler.close()
    except Exception as e:
        latest={'status':'error','generated_at_ms':int(time.time()*1000),'message':'The simulation stopped. No live data is available.'}
        import traceback;traceback.print_exc()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ['/state','/health']:
            self.send_error(404);return
        data=latest if self.path=='/state' else {k:latest.get(k) for k in ['status','run_id','sequence','generated_at_ms']}
        body=json.dumps(data,separators=(',',':')).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
    def log_message(self,*args):pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8766)
    p.add_argument('--decoder',choices=['biological','bci'],default='bci')
    p.add_argument('--seed',type=int,default=41027);p.add_argument('--reward',choices=['off','sugar'],default='off')
    p.add_argument('--condition',choices=['intact','blank_vision','frozen_vision','retina_disconnected','all_edges_disconnected','controls_clamped'],default='intact')
    args=p.parse_args();worker=threading.Thread(target=run_loop,args=(args,),daemon=True);worker.start()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:stop.set();server.server_close();worker.join(timeout=15)
if __name__=='__main__':main()
