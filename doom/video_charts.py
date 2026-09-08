"""Charts of the learning telemetry recorded next to a video archive.

Reads `video-metrics.jsonl` from a run directory written by doom.video and
draws the learning curve (changed edges, efficacy statistics, delivered
reinforcement), the latest efficacy histogram, round survival and simulation
speed. Every figure is labelled as an unvalidated experiment; there is no
"percent learned" score because none is defined. Optionally renders a
learning-curve movie (one frame per minute of neural time) with ffmpeg.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np

LABEL='TRAINING / EXPERIMENTAL / UNVALIDATED - no learning score is defined'


def load(run_dir):
    rows=[json.loads(l) for l in (Path(run_dir)/'video-metrics.jsonl').read_text().splitlines() if l.strip()]
    frames=[r for r in rows if 'archive_file' in r];events=[r for r in rows if 'event' in r]
    return frames,events


def series(frames):
    t=np.asarray([f['neural_ms']/1000 for f in frames]);wall=np.asarray([f['wall_s'] for f in frames]);speed=np.asarray([f['sim_speed'] for f in frames])
    learn=[f.get('learning') or {} for f in frames]
    def col(k):return np.asarray([l.get(k,np.nan) if l else np.nan for l in learn],dtype=float)
    return {'neural_s':t,'wall_s':wall,'speed':speed,'changed':col('changed_edges'),'mean':col('mean_efficacy'),'min':col('minimum_efficacy'),
            'max':col('maximum_efficacy'),'delivered_s':col('delivered_ms')/1000,'damage':col('damage_events'),'health':np.asarray([f['game'].get('health',np.nan) for f in frames],dtype=float)}


def survival(events):
    return [(e['tick'],e['detail'].get('survived_ticks',0)/35) for e in events if e['event']=='round_end']


def draw_all(run_dir,out_dir=None,upto=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run_dir=Path(run_dir);out=Path(out_dir) if out_dir else run_dir/'charts';out.mkdir(parents=True,exist_ok=True)
    frames,events=load(run_dir)
    if upto is not None:frames=[f for f in frames if f['neural_ms']/1000<=upto]
    if not frames:raise SystemExit('No frames in video-metrics.jsonl')
    s=series(frames);has_learning=np.isfinite(s['changed']).any()
    fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True,dpi=110)
    if has_learning:
        axes[0].plot(s['neural_s'],s['changed'],color='#5aaaff');axes[0].set_ylabel('changed edges (of 4,184)')
        axes[1].plot(s['neural_s'],s['mean'],label='mean',color='#78dc8c');axes[1].plot(s['neural_s'],s['min'],label='min',color='#c8b450');axes[1].plot(s['neural_s'],s['max'],label='max',color='#ff7850')
        axes[1].set_ylabel('efficacy / baseline');axes[1].legend(loc='upper right')
        axes[2].plot(s['neural_s'],s['delivered_s'],color='#ff7850');axes[2].set_ylabel('PPL101 reinforcement delivered (s)')
    else:
        for ax in axes[:3]:ax.text(.5,.5,'no learning telemetry (fixed weights)',ha='center',transform=ax.transAxes)
    axes[3].plot(s['neural_s'],s['speed'],color='#8c8c96');axes[3].set_ylabel('sim speed (x realtime)');axes[3].set_xlabel('neural time (s)')
    fig.suptitle(f'Doomfly learning curve - {LABEL}',fontsize=10);fig.tight_layout();fig.savefig(out/'learning-curve.png');plt.close(fig)
    last=next((f['learning'] for f in reversed(frames) if f.get('learning')),None)
    if last and last.get('efficacy_histogram'):
        fig,ax=plt.subplots(figsize=(8,3.5),dpi=110);h=last['efficacy_histogram'];lo,hi=last.get('histogram_range',[0.1,2.0]) if 'histogram_range' in last else (0.1,2.0)
        ax.bar(np.linspace(lo,hi,len(h),endpoint=False),h,width=(hi-lo)/len(h),align='edge',color='#78dc8c');ax.set_xlabel('efficacy / baseline');ax.set_ylabel('plastic edges')
        ax.set_title(f'Latest efficacy histogram at {s["neural_s"][-1]:.0f} s neural time - {LABEL}',fontsize=9);fig.tight_layout();fig.savefig(out/'histogram-latest.png');plt.close(fig)
    rounds=survival(events)
    fig,ax=plt.subplots(figsize=(8,3.5),dpi=110)
    if rounds:ax.bar(range(len(rounds)),[r[1] for r in rounds],color='#c8b450');ax.set_xlabel('round (in this archive)');ax.set_ylabel('survival (s)')
    else:ax.text(.5,.5,'no completed rounds in this archive',ha='center',transform=ax.transAxes)
    ax.set_title(f'Round survival - {LABEL}',fontsize=9);fig.tight_layout();fig.savefig(out/'survival.png');plt.close(fig)
    summary={'frames':len(frames),'neural_seconds':[float(s['neural_s'][0]),float(s['neural_s'][-1])],'rounds_completed':len(rounds),
             'mean_survival_s':float(np.mean([r[1] for r in rounds])) if rounds else None,'final_learning':last,'label':LABEL,
             'events':{k:sum(1 for e in events if e['event']==k) for k in sorted({e['event'] for e in events})}}
    (out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    return out,summary


def movie(run_dir,out_dir=None,minutes_per_frame=1.0,fps=10,encoder='x264'):
    """One chart frame per `minutes_per_frame` of neural time, encoded to learning-curve.mp4 and .gif."""
    run_dir=Path(run_dir);out=Path(out_dir) if out_dir else run_dir/'charts';frames_dir=out/'movie';frames_dir.mkdir(parents=True,exist_ok=True)
    frames,_=load(run_dir);end=frames[-1]['neural_ms']/1000;start=frames[0]['neural_ms']/1000
    stops=np.arange(start+60*minutes_per_frame,end+1e-9,60*minutes_per_frame)
    if len(stops)==0:stops=np.asarray([end])
    for i,upto in enumerate(stops):
        draw_all(run_dir,frames_dir/f'tmp{i:05d}',upto=float(upto))
        (frames_dir/f'tmp{i:05d}'/'learning-curve.png').replace(frames_dir/f'frame-{i:05d}.png')
    from doom.video.ffmpeg import encoder_args
    _,codec=encoder_args(encoder,device='/dev/dri/renderD128',rate_control='archive') if encoder!='x264' else ([],['-c:v','libx264','-pix_fmt','yuv420p','-crf','20'])
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(frames_dir/'frame-%05d.png'),'-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2',*codec,str(out/'learning-curve.mp4')],check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(frames_dir/'frame-%05d.png'),'-vf','scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',str(out/'learning-curve.gif')],check=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,help='outputs/doom/video/<run_id>')
    p.add_argument('--out');p.add_argument('--movie',action='store_true');p.add_argument('--minutes-per-frame',type=float,default=1.0)
    a=p.parse_args()
    out,summary=draw_all(a.run,a.out)
    if a.movie:movie(a.run,a.out,a.minutes_per_frame)
    print(json.dumps({'charts':str(out),**{k:v for k,v in summary.items() if k!='final_learning'}},indent=1))


if __name__=='__main__':main()
