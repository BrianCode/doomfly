"""Cut clips around recorded events from a neural-time video archive.

Frames are located through video-metrics.jsonl (frame index -> archive file and
frame offset), so a clip is addressed by game tick. Segments start on forced
keyframes; `-c copy` cuts are keyframe-aligned (up to 2 s early), `--exact`
re-encodes for frame-accurate bounds. Each clip keeps its provenance in
clips/index.json (run, study, tick range, neural seconds, captured wall speed,
first/last source frame digests) and is labelled a replay at 1x brain time.
"""
import argparse,json,subprocess
from pathlib import Path

ARCHIVE_FPS=35


def load(run_dir):
    rows=[json.loads(l) for l in (Path(run_dir)/'video-metrics.jsonl').read_text().splitlines() if l.strip()]
    return [r for r in rows if 'archive_file' in r],[r for r in rows if 'event' in r]


def locate(frames,tick):
    """Return the metrics row for `tick` (or the nearest later recorded tick)."""
    for f in frames:
        if f['tick']>=tick:return f
    return None


def clip_ranges(frames,events,kinds,before_s,after_s,ticks=()):
    targets=[(e['event'],e['tick']) for e in events if e['event'] in kinds]+[('tick',t) for t in ticks]
    out=[]
    for kind,tick in targets:
        centre=locate(frames,tick)
        if centre is None:continue
        first=max(frames[0]['frame_index'],centre['frame_index']-round(before_s*ARCHIVE_FPS))
        last=min(frames[-1]['frame_index'],centre['frame_index']+round(after_s*ARCHIVE_FPS))
        a=frames[first-frames[0]['frame_index']];b=frames[last-frames[0]['frame_index']]
        if a['archive_file']!=b['archive_file']:
            b=frames[[f['archive_file'] for f in frames].index(a['archive_file'])+sum(1 for f in frames if f['archive_file']==a['archive_file'])-1]
        out.append({'event':kind,'tick':tick,'file':a['archive_file'],'start_s':a['archive_frame']/ARCHIVE_FPS,'end_s':(b['archive_frame']+1)/ARCHIVE_FPS,
                    'tick_range':[a['tick'],b['tick']],'frame_range':[a['frame_index'],b['frame_index']],
                    'neural_seconds':[a['neural_ms']/1000,b['neural_ms']/1000],'captured_wall_speed':[a['sim_speed'],b['sim_speed']],
                    'source_frame_sha256_first':a['source_frame_sha256'],'source_frame_sha256_last':b['source_frame_sha256']})
    return out


def cut(run_dir,clips,exact=False,out_dir=None):
    run_dir=Path(run_dir);out=Path(out_dir) if out_dir else run_dir/'clips';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run_dir/'video-manifest.json').read_text())
    index=[]
    for c in clips:
        name=f"{c['event']}-{c['tick']}.mp4"
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',f"{c['start_s']:.3f}",'-to',f"{c['end_s']:.3f}",'-i',str(run_dir/c['file'])]
        cmd+=['-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p'] if exact else ['-c','copy']
        cmd+=['-movflags','+faststart',str(out/name)]
        subprocess.run(cmd,check=True)
        index.append({**c,'clip':name,'exact':exact,'run_id':manifest['run_id'],'study_id':manifest.get('study_id'),'phase':manifest.get('phase'),
                      'label':'REPLAY at 1x brain time; overlay is telemetry, not neural input','model_revision':manifest.get('model_revision')})
    (out/'index.json').write_text(json.dumps(index,indent=1)+'\n')
    return out,index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--events',default='damage,kill,round_end');p.add_argument('--tick',type=int,action='append',default=[])
    p.add_argument('--before',type=float,default=3);p.add_argument('--after',type=float,default=5);p.add_argument('--exact',action='store_true')
    p.add_argument('--limit',type=int,default=20);p.add_argument('--out')
    a=p.parse_args()
    frames,events=load(a.run)
    clips=clip_ranges(frames,events,set(a.events.split(',')) if a.events else set(),a.before,a.after,a.tick)[:a.limit]
    out,index=cut(a.run,clips,a.exact,a.out)
    print(json.dumps({'clips':len(index),'directory':str(out)}))


if __name__=='__main__':main()
