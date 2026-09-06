"""Causal and numerical checks; these do not establish biological validity."""
import numpy as np
import pytest
from doom_learning_v2.brain import MemoryBrain


def fixture_brain(tmp_path, eta=.001, modulation=True):
    p=tmp_path/'memory.npz';n=6
    np.savez(p,ptr=np.array([0,1,1,2,2,3,4],dtype=np.int64),post=np.array([1,3,1,3],dtype=np.int32),
        weight=np.array([20,20,.275,.275],dtype=np.float32),ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.array([0,2],dtype=np.int32),sugar=np.empty(0,dtype=np.int32),superclass=np.array(['test']*n))
    c={'edges':np.array([0,1],dtype=np.int64),'pre':np.array([0,2],dtype=np.int32),
       'kc_mask':np.array([1,0,1,0,0,0],dtype=np.uint8),
       'dan_index':np.array([-1,-1,-1,-1,0,1] if modulation else [-1]*n,dtype=np.int8),
       'gain':np.array([[1,0],[0,1]],dtype=np.float32),'kc':np.array([0,2]),'mb':np.array([1,3]),'dan':np.array([4,5])}
    return MemoryBrain(p,eta=eta,circuit=c,modulation_mask=c["dan_index"]>=0)


def test_no_dopamine_no_plasticity_and_wrong_compartment_unchanged(tmp_path):
    b=fixture_brain(tmp_path);before=b.weight.copy()
    b.step([],100,learning=True,stimulation=([0],30),lamina_bias=0)
    np.testing.assert_array_equal(before,b.weight)
    b.step([],100,learning=True,stimulation=([0,4],30),lamina_bias=0)
    assert b.weight[0]<before[0]
    np.testing.assert_array_equal(before[1:],b.weight[1:])


def test_pairing_order_and_frozen_control(tmp_path):
    b=fixture_brain(tmp_path);before=b.weight.copy()
    b.step([],100,learning=True,stimulation=([4],30),lamina_bias=0)
    b.step([],500,learning=True,lamina_bias=0)
    b.step([],100,learning=True,stimulation=([0],30),lamina_bias=0)
    np.testing.assert_array_equal(before,b.weight)
    b.reset();b.step([],100,learning=False,stimulation=([0,4],30),lamina_bias=0)
    np.testing.assert_array_equal(before,b.weight)


def test_memory_checkpoint_reproduces_continuation(tmp_path):
    b=fixture_brain(tmp_path)
    b.step([],100,learning=True,stimulation=([0,4],30),lamina_bias=0)
    p=tmp_path/'checkpoint.npz';b.checkpoint(p)
    c1,_=b.step([],50,learning=True,stimulation=([0,4],30),lamina_bias=0)
    expected={k:getattr(b,k).copy() for k in ['weight',*b.fields]}
    b.restore(p);c2,_=b.step([],50,learning=True,stimulation=([0,4],30),lamina_bias=0)
    np.testing.assert_array_equal(c1,c2)
    for k,v in expected.items():np.testing.assert_array_equal(v,getattr(b,k),err_msg=k)
    b.eta=.002
    with pytest.raises(ValueError,match='provenance'):b.restore(p)


def test_memory_only_persists_and_reset_erases_changes(tmp_path):
    b=fixture_brain(tmp_path)
    b.step([],100,learning=True,stimulation=([0,4],30),lamina_bias=0)
    learned=b.weight.copy();assert b.memory()['changed_edges']==1
    b.reset(keep_memory=True);np.testing.assert_array_equal(learned,b.weight)
    assert not b.eligibility.any() and not b.modulation.any()
    b.reset();np.testing.assert_array_equal(b.weight[:2],b.baseline_plastic)


def test_unchanged_integration_matches_original_kernel(tmp_path):
    from doom.native import NativeBrain
    b=fixture_brain(tmp_path,modulation=False)
    a=NativeBrain(tmp_path/'memory.npz')
    for bias in [12.,0.,18.,3.]:
        ca,_=a.step(np.empty(0),60,lamina_bias=bias)
        cb,_=b.step([],60,lamina_bias=bias)
        np.testing.assert_array_equal(ca,cb)
        np.testing.assert_array_equal(a.v,b.v)
        np.testing.assert_array_equal(a.g,b.g)


def test_modulation_does_not_act_as_fast_excitation(tmp_path):
    b=fixture_brain(tmp_path)
    b.step([],100,learning=False,stimulation=([4],30),lamina_bias=0)
    assert b.counts[4]>0 and b.modulation[1]>0
    assert b.v[1]==-52 and b.g[1]==0 and b.counts[1]==0


def test_ltd_matches_independent_per_tick_event_rule(tmp_path):
    b=fixture_brain(tmp_path,eta=.01)
    expected=20.;elig=0.;dopamine_arrivals={}
    for tick in range(800):
        c,_=b.step([],.1,learning=True,stimulation=([0,4],30),lamina_bias=0)
        elig=elig*np.exp(-.1/1000)+c[0]
        dopamine_arrivals[tick+18]=int(c[4])
        for _ in range(dopamine_arrivals.get(tick,0)):
            expected=float(np.float32(max(2.,expected*np.exp(-.01*elig))))
        assert b.weight[0]==pytest.approx(expected,abs=2e-5)


def test_checkpoint_rejects_different_compartment_mapping(tmp_path):
    b=fixture_brain(tmp_path);p=tmp_path/'checkpoint.npz';b.checkpoint(p)
    b.circuit['gain'][0,0]=.5
    with pytest.raises(ValueError,match='provenance'):b.restore(p)


def test_checkpoint_rejects_different_original_weights(tmp_path):
    b=fixture_brain(tmp_path);p=tmp_path/'checkpoint.npz';b.checkpoint(p)
    graph=tmp_path/'memory.npz'
    with np.load(graph) as data:arrays={k:data[k] for k in data.files}
    arrays['weight'][0]=21.;np.savez(graph,**arrays)
    c=MemoryBrain(graph,circuit=b.circuit,modulation_mask=b.modulation_mask)
    with pytest.raises(ValueError,match='provenance'):c.restore(p)
