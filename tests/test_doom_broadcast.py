"""Transport-only fixtures, never neural behavior or learning data."""
import copy
import json
from doom.broadcast import Broadcast, RETAIN_SEGMENTS


def frame(sequence=1,second=1788650000):
    return {'schema':1,'status':'running','run_id':'a'*36,'condition':'intact','decoder':'bci',
        'manifest':{},'protocol':{},'reward':{'plasticity':False},'sequence':sequence,
        'generated_at_ms':second*1000,'frame':'data:image/jpeg;base64,AA==',
        'clocks':{},'game':{},'episodes':[],'action':{},'readouts':[],
        'total_spikes':12,'window_spikes':3,'window_ms':28.6,'total_action_ticks':1,
        'audit':{'source_frame_sha256':'actual-test-input-hash'},
        'retina':{'uv':[[0,0]],'full_sample_count':1,'display_stride':8,'luminance':[.25]},
        'raster':{'neuron_ids':['1'],'bins':[{'neural_ms':28.6,'window_ms':28.6,'counts':[3]}]}}


def test_segments_preserve_frame_timing_audit_and_bytes():
    stream=Broadcast();a=frame();original=copy.deepcopy(a)
    stream.publish(a)
    assert json.loads(stream.index)['segments']==[]
    stream.publish(frame(2,1788650001))
    packet=stream.get_segment('a'*36,'1788650000')
    view=json.loads(packet)['frames'][0]
    assert view['frame']==a['frame'] and view['audit']==a['audit']
    assert view['generated_at_ms']==a['generated_at_ms'] and view['luminance']==[.25]
    assert a==original
    stream.publish(frame(3,1788650002))
    assert stream.get_segment('a'*36,'1788650000')==packet


def test_training_phase_and_live_memory_are_transmitted_per_frame():
    stream=Broadcast();a=frame();a['protocol']={'phase':'training'}
    a['reward']['plasticity']=True;a['learning']={'changed_edges':12,'validated':False}
    stream.publish(a)
    b=frame(2,1788650001);b['protocol']=a['protocol'];stream.publish(b)
    packet=json.loads(stream.get_segment('a'*36,'1788650000'))
    assert packet['frames'][0]['learning']==a['learning']
    assert packet['frames'][0]['reward']['plasticity']
    assert json.loads(stream.index)['phase']=='training'


def test_bounds_and_run_isolation():
    stream=Broadcast()
    for i in range(RETAIN_SEGMENTS+5):stream.publish(frame(i+1,1788650000+i))
    assert len(stream.segments)==RETAIN_SEGMENTS
    assert stream.get_segment('a'*36,'1788650000') is None
    assert len(json.loads(stream.index)['segments'])==4
    new=frame();new['run_id']='b'*36;stream.publish(new)
    assert not stream.segments and stream.get_segment('a'*36,'1788650004') is None


def test_offline_never_advertises_recorded_frames_as_live():
    stream=Broadcast();stream.publish(frame());stream.publish(frame(2,1788650001))
    stream.offline({'status':'error','generated_at_ms':1788650002000})
    assert json.loads(stream.index)['status']=='error'
    assert json.loads(stream.state)['status']=='error'
