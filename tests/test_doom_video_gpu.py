"""Hardware encoder smoke tests (run with DOOMFLY_GPU_TESTS=1 on a host with ffmpeg + VAAPI/QSV/NVENC)."""
import json,os,shutil,subprocess
import numpy as np
import pytest
from doom.video.ffmpeg import FfmpegSink,build_archive_cmd,probe_encoders

pytestmark=pytest.mark.skipif(os.environ.get('DOOMFLY_GPU_TESTS')!='1' or shutil.which('ffprobe') is None,reason='set DOOMFLY_GPU_TESTS=1 with ffmpeg installed')
WORKING=probe_encoders(['vaapi','qsv','nvenc']) if os.environ.get('DOOMFLY_GPU_TESTS')=='1' else []

@pytest.mark.parametrize('encoder',WORKING or ['none'])
def test_hardware_encoder_writes_a_35fps_archive(tmp_path,encoder):
    if encoder=='none':pytest.skip('no hardware encoder passed the probe')
    w,h=1280,720
    sink=FfmpegSink(lambda start:build_archive_cmd(encoder,width=w,height=h,fps=35,out_dir=tmp_path,segment_seconds=300),name='archive',frame_bytes=w*h*3,log_path=tmp_path/'ffmpeg.log')
    rng=np.random.default_rng(0)
    for i in range(100):
        frame=np.full((h,w,3),i*2,np.uint8);frame[100:200,100:400]=rng.integers(0,255,(100,300,3),dtype=np.uint8)
        assert sink.write(frame.tobytes()),sink.last_error
    sink.close()
    out=tmp_path/'archive-00000.mp4';assert out.exists(),(tmp_path/'ffmpeg.log').read_text()
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v','-show_entries','stream=codec_name,r_frame_rate,nb_read_frames,width','-of','json',str(out)],capture_output=True,text=True)
    s=json.loads(probe.stdout)['streams'][0]
    assert s['codec_name']=='h264' and s['r_frame_rate']=='35/1' and int(s['width'])==w and int(s['nb_read_frames'])>=98
