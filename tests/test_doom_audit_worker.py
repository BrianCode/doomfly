import hashlib,json,logging,threading,time
import numpy as np
import pytest
from doom.audit_worker import AuditWorker,AuditHandle,DIGESTS

class Capture(logging.Handler):
    def __init__(self,gate=None):
        super().__init__();self.lines=[];self.gate=gate;self.entered=threading.Event()
    def emit(self,record):
        self.entered.set()
        if self.gate is not None:self.gate.wait()
        self.lines.append(record.getMessage())

def make_logger(name,handler):
    logger=logging.getLogger(name);logger.handlers=[handler];logger.setLevel(logging.INFO);logger.propagate=False
    return logger

def arrays(rng):
    light=rng.random(3300,dtype=np.float32);frame=rng.integers(0,256,(480,640,3),dtype=np.uint8);counts=rng.integers(0,3,166000).astype(np.int32)
    return light,frame,counts

def inline_event(event,light,frame,counts):
    return {**event,'input_sha256':hashlib.sha256(light.tobytes()).hexdigest(),
        'source_frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),
        'spike_counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest()}

def test_digests_match_inline_hashlib_and_keep_key_order():
    capture=Capture();worker=AuditWorker(make_logger('audit-test-digests',capture),maxsize=4)
    rng=np.random.default_rng(1);light,frame,counts=arrays(rng)
    event={'tick':1,'game':{'health':100},'input_sha256':None,'source_frame_sha256':None,'spike_counts_sha256':None,'applied':{'turn':0.}}
    expected=inline_event({k:v for k,v in event.items() if k not in DIGESTS},light,frame,counts)
    expected={k:expected[k] for k in event}
    handle=worker.submit(event,light,frame,counts)
    assert isinstance(handle,AuditHandle)
    done=handle.result(timeout=5)
    assert done is event and done==expected and list(done)==list(expected)
    worker.close(timeout=5)
    assert capture.lines==[json.dumps(expected,separators=(',',':'))]
    assert worker.stats()=={'submitted':1,'written':1,'blocked':0,'depth':0,'maxsize':4,'failed':None}

def test_lines_keep_submission_order_under_concurrent_submissions():
    capture=Capture();worker=AuditWorker(make_logger('audit-test-order',capture),maxsize=3)
    rng=np.random.default_rng(2);light,frame,counts=arrays(rng);lock=threading.Lock();order=[]
    def producer(name):
        for i in range(40):
            with lock:
                event={'producer':name,'seq':i,'n':len(order)};order.append((name,i))
                worker.submit(event,light,frame,counts)
    threads=[threading.Thread(target=producer,args=(n,)) for n in 'abc']
    for t in threads:t.start()
    for t in threads:t.join()
    worker.close(timeout=10)
    written=[json.loads(l) for l in capture.lines]
    assert [(w['producer'],w['seq']) for w in written]==order
    assert [w['n'] for w in written]==list(range(120))
    assert worker.stats()['written']==120 and worker.stats()['submitted']==120

def test_flush_drains_everything():
    capture=Capture();worker=AuditWorker(make_logger('audit-test-flush',capture),maxsize=2)
    rng=np.random.default_rng(3);light,frame,counts=arrays(rng)
    handles=[worker.submit({'tick':i},light,frame,counts) for i in range(25)]
    worker.flush()
    assert len(capture.lines)==25 and all(h.done.is_set() for h in handles)
    assert [json.loads(l)['tick'] for l in capture.lines]==list(range(25))
    assert worker.stats()['depth']==0 and worker.stats()['blocked']>0
    worker.close(timeout=5)

def test_stalled_consumer_blocks_rather_than_drops():
    gate=threading.Event();capture=Capture(gate);worker=AuditWorker(make_logger('audit-test-stall',capture),maxsize=2)
    rng=np.random.default_rng(4);light,frame,counts=arrays(rng)
    try:
        first=worker.submit({'tick':0},light,frame,counts)
        assert capture.entered.wait(5) and not first.done.is_set()   # worker holds tick 0 in the stalled handler
        worker.submit({'tick':1},light,frame,counts);worker.submit({'tick':2},light,frame,counts)
        assert worker.queue.full() and worker.blocked==0
        returned=threading.Event()
        def blocked_submit():worker.submit({'tick':3},light,frame,counts);returned.set()
        t=threading.Thread(target=blocked_submit,daemon=True);t.start()
        assert not returned.wait(.3)
        assert worker.blocked==1 and capture.lines==[]
    finally:gate.set()                            # never leave the handler lock held at interpreter exit
    t.join(5);worker.close(timeout=5)
    assert [json.loads(l)['tick'] for l in capture.lines]==[0,1,2,3]
    assert worker.stats()['blocked']==1 and worker.stats()['written']==4

def test_handler_failure_surfaces_on_next_submit():
    class Broken(logging.Handler):
        def emit(self,record):raise RuntimeError('disk full')
    logger=make_logger('audit-test-broken',Broken())
    worker=AuditWorker(logger,maxsize=2);rng=np.random.default_rng(5);light,frame,counts=arrays(rng)
    handle=worker.submit({'tick':0},light,frame,counts)
    with pytest.raises(RuntimeError,match='disk full'):handle.result(timeout=5)
    with pytest.raises(RuntimeError,match='disk full'):worker.submit({'tick':1},light,frame,counts)
    with pytest.raises(RuntimeError,match='disk full'):worker.close(timeout=5)
