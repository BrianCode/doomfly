import numpy as np
import pytest
from doom_learning_v6.brain import MemoryBrain


def brain(tmp_path):
    p=tmp_path/'graph.npz';n=4
    np.savez(p,ptr=np.array([0,1,1,2,2],dtype=np.int64),post=np.array([1,1],dtype=np.int32),
        weight=np.array([20.,.275],dtype=np.float32),ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),lamina=np.empty(0,dtype=np.int32),
        sugar=np.empty(0,dtype=np.int32),superclass=np.array(['test']*n))
    c={'edges':np.array([0],dtype=np.int64),'pre':np.array([0],dtype=np.int32),
       'kc_mask':np.array([1,0,0,0],dtype=np.uint8),'dan_index':np.array([-1,-1,0,-1],dtype=np.int8),
       'gain':np.array([[1.]],dtype=np.float32),'kc':np.array([0]),'mb':np.array([1]),'dan':np.array([2])}
    return MemoryBrain(p,eta=.001,circuit=c,modulation_mask=np.array([0,0,1,0]))


def test_full_state_checkpoint_reproduces_ongoing_memory(tmp_path):
    b=brain(tmp_path)
    b.step([],100,learning=True,stimulation=([0],20),lamina_bias=0)
    b.step([],100,learning=True,stimulation=([2],12),lamina_bias=0)
    assert b.memory_u[0]<0 and b.weight[0]<20
    p=tmp_path/'checkpoint.npz';b.checkpoint(p)
    c,_=b.step([],200,learning=True,stimulation=([0,2],20),lamina_bias=0)
    expected={k:getattr(b,k).copy() for k in ['weight',*b.fields]}
    b.restore(p);d,_=b.step([],200,learning=True,stimulation=([0,2],20),lamina_bias=0)
    np.testing.assert_array_equal(c,d)
    for k,v in expected.items():np.testing.assert_array_equal(v,getattr(b,k),err_msg=k)


def test_only_memory_persists_when_fast_state_resets(tmp_path):
    b=brain(tmp_path);b.memory_u[:]=-.2;b.memory_w[:]=-.1;b.weight[0]=18
    b.rate_kc[:]=5;b.rate_dan[:]=5;b.reset(keep_memory=True)
    assert b.memory_u[0]==-.2 and b.memory_w[0]==-.1 and b.weight[0]==18
    assert not b.rate_kc.any() and not b.rate_dan.any()
    b.reset();assert b.weight[0]==20 and not b.memory_u.any() and not b.memory_w.any()


def test_freeze_mode_and_background_are_checkpointed(tmp_path):
    b=brain(tmp_path);b.weights_frozen=True;b.tonic[0]=12
    b.step([],100,learning=True,stimulation=([2],12),lamina_bias=0)
    assert b.weight[0]==20
    p=tmp_path/'checkpoint.npz';b.checkpoint(p)
    b.weights_frozen=False;b.restore(p);assert b.weights_frozen
    b.tonic[0]=11
    with pytest.raises(ValueError,match='provenance'):b.restore(p)


def test_no_learning_preserves_original_neural_kernel(tmp_path):
    from doom_learning_v4.brain import MemoryBrain as Previous
    b=brain(tmp_path);c=Previous(tmp_path/'graph.npz',circuit=b.circuit,modulation_mask=b.modulation_mask)
    for stimulus in [([0],20),([2],20),([0,2],20)]:
        a,_=b.step([],100,stimulation=stimulus,lamina_bias=0)
        d,_=c.step([],100,stimulation=stimulus,lamina_bias=0)
        np.testing.assert_array_equal(a,d)
        np.testing.assert_allclose(b.v,c.v,atol=.002,rtol=0)
        np.testing.assert_array_equal(b.weight,c.weight)
