"""Video export: honest frame handling, non-blocking hand-off, encoder plumbing (no GPU needed)."""
import json,shutil,subprocess,time
import numpy as np
import pytest
from doom.video import VideoTick
from doom.video.compose import Compositor,STRIP_HEIGHT
from doom.video.ring import FrameRing
from doom.video.pacer import LivePacer
from doom.video.metrics import MetricsWriter
from doom.video.ffmpeg import build_archive_cmd,build_live_cmd,redacted_command,encoder_args,FfmpegSink

def tick(i,frame,n=64,learning=None,health=100,kills=0,finished=False):
    rng=np.random.default_rng(i)
    return VideoTick(tick=i,frame=frame,counts=rng.integers(0,3,n).astype(np.int32),game={'health':health,'kills':kills,'ammo':50,'enemies':3,'tick':i,'finished':finished},
        action={'turn':0.5,'forward':1.0,'attack':False},neural_ms=i*28.6,wall_s=i*0.03,brain_step_ms=12.3,sim_speed=0.95,
        source_frame_sha256='0'*64,episode=1,learning=learning,run_id='run-test',study_id='study-test',phase='training' if learning else 'baseline')

def test_compositor_game_region_is_a_nearest_neighbour_resize_of_one_frame():
    frame=np.random.default_rng(1).integers(0,255,(48,64,3),dtype=np.uint8)
    c=Compositor(96+320,72,groups=[np.arange(0,32),np.arange(32,64)],group_names=['a','b'],display=np.arange(16),overlays=('bars','raster'))
    out=c.render(tick(1,frame))
    assert out.shape==(72,96+320,3) and out.dtype==np.uint8
    ys=(np.arange(72)*48//72);xs=(np.arange(96)*64//96)
    expected=frame[ys][:,xs]
    # the status strip overlays the bottom rows of the game view; everything above it is the exact resize
    np.testing.assert_array_equal(out[:72-STRIP_HEIGHT,:96],expected[:72-STRIP_HEIGHT])
    assert c.frames==1

def test_compositor_renders_learning_and_no_learning_layouts():
    frame=np.zeros((480,640,3),np.uint8)
    c=Compositor(1280,720,groups=[np.arange(64)],group_names=['x'],display=np.arange(32),phase='training')
    learning={'plastic_edges':4184,'changed_edges':12,'mean_efficacy':.99,'minimum_efficacy':.5,'maximum_efficacy':1.2,'efficacy_histogram':[0]*19+[4184],
              'histogram_range':[0.1,2.0],'damage_events':2,'delivered_ms':400,'stimulus_active':True}
    a=c.render(tick(1,frame,learning=learning)).copy(); b=c.render(tick(2,frame,learning=None)).copy()
    assert a.shape==b.shape==(720,1280,3) and a.any() and b.any()
    assert not np.array_equal(a[:,960:],b[:,960:])     # the panels differ between the two layouts

def test_compositor_raster_scrolls_with_recorded_counts():
    c=Compositor(1280,720,display=np.arange(8),overlays=('raster',))
    counts=np.zeros(64,np.int32);counts[3]=6
    c.push_raster(counts,28.6)
    assert c.raster[3,-1]>0 and c.raster[:,-2].sum()==0
    c.push_raster(np.zeros(64,np.int32),28.6)
    assert c.raster[3,-2]>0 and c.raster[3,-1]==0

def test_ring_never_blocks_and_records_drops_as_tick_gaps():
    r=FrameRing(capacity=3)
    for i in range(1,6):r.offer(tick(i,None))
    items=r.drain(timeout=0)
    assert [t.tick for t in items]==[3,4,5]
    s=r.stats();assert s['dropped']==2 and s['offered']==5 and s['dropped_tick_ranges']==[[1,2]]
    t0=time.perf_counter()
    for i in range(1000):r.offer(tick(i,None))
    assert time.perf_counter()-t0<0.5

def test_pacer_counts_held_and_skipped_frames():
    now=[0.0];p=LivePacer(fps=30,clock=lambda:now[0])
    # 35 produced per second, 30 shown: 5 skipped
    shown=0
    for i in range(35):
        now[0]=i/35;p.submit(f'f{i}',i)
        while p.due(now[0]):
            p.take(now[0]);shown+=1
    assert p.skipped>=4 and p.held==0 and shown in (29,30,31)
    # 8 produced per second, 30 shown: about 22 held
    q=LivePacer(fps=30,clock=lambda:now[0]);now[0]=0.0;held=0;shown=0
    for i in range(8):
        now[0]=i/8;q.submit(f'g{i}',i)
        while q.due(now[0]):
            _,h=q.take(now[0]);held+=h;shown+=1
    assert 17<=held<=23 and q.stats()['held']==held and shown==held+8

def test_metrics_and_manifest_files(tmp_path):
    m=MetricsWriter(tmp_path,{'run_id':'r'})
    m.frame({'frame_index':0,'tick':5});m.event('damage',5,0,{'health':90});m.close(frames=1)
    rows=[json.loads(l) for l in (tmp_path/'video-metrics.jsonl').read_text().splitlines()]
    assert rows[0]['tick']==5 and rows[1]['event']=='damage'
    man=json.loads((tmp_path/'video-manifest.json').read_text());assert man['frames']==1 and man['run_id']=='r'

def test_ffmpeg_commands_are_correct_and_redacted(tmp_path):
    cmd=build_archive_cmd('vaapi',width=1280,height=720,fps=35,out_dir=tmp_path,device='/dev/dri/renderD128',segment_seconds=300)
    assert 'h264_vaapi' in cmd and '-vaapi_device' in cmd and cmd[cmd.index('-segment_time')+1]=='300' and cmd[-1].endswith('archive-%05d.mp4')
    assert 'expr:gte(t,n_forced*300)' in cmd
    for enc in ['qsv','nvenc','x264']:
        pre,codec=encoder_args(enc,device='/dev/dri/renderD128',rate_control='live');assert '-c:v' in codec
    live=build_live_cmd('nvenc',width=1280,height=720,fps=30,rtmp_url='rtmp://live.twitch.tv/app/live_123_SECRETKEY',hls_dir=tmp_path)
    red=redacted_command(live)
    assert not any('SECRETKEY' in part for part in red) and any('<redacted>' in part for part in red)
    assert '-use_fifo' in live and 'anullsrc' in ' '.join(live) and 'hls_time=2' in live[-1]

@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,reason='ffmpeg not installed')
def test_ffmpeg_sink_writes_an_mp4_with_x264(tmp_path):
    w,h=64,48
    sink=FfmpegSink(lambda start:build_archive_cmd('x264',width=w,height=h,fps=35,out_dir=tmp_path,segment_seconds=300,start_number=0),
        name='archive',frame_bytes=w*h*3,log_path=tmp_path/'ffmpeg.log')
    frame=np.zeros((h,w,3),np.uint8)
    for i in range(70):
        frame[:]=i*3;assert sink.write(frame.tobytes())
    sink.close()
    out=tmp_path/'archive-00000.mp4';assert out.exists() and sink.frames_written==70
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v','-show_entries','stream=codec_name,r_frame_rate,nb_read_frames','-of','json',str(out)],capture_output=True,text=True)
    s=json.loads(probe.stdout)['streams'][0]
    assert s['codec_name']=='h264' and s['r_frame_rate']=='35/1' and int(s['nb_read_frames'])==70
