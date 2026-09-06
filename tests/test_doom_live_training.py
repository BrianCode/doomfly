import json
from types import SimpleNamespace
import numpy as np
import pytest
from doom.training import DamageTraining
from doom.training_checkpoint import TrainingCheckpoints
from doom.engine import NeuralControls


class PulseBrain:
    def __init__(self):
        self.n=3;self.cursor=0;self.circuit={'dan':np.array([2])}
        self.calls=[];self.counts=np.zeros(3,dtype=np.int32)
    def rgb_step(self,rgb,ms,learning,stimulation):
        n=round(ms*10);self.calls.append((self.cursor,n,learning,stimulation is not None))
        self.cursor+=n
        return np.array([n,0,n if stimulation else 0],dtype=np.int32),0.


def test_damage_is_delayed_exact_and_never_selects_a_control():
    b=PulseBrain();t=DamageTraining(b);rgb=np.zeros((2,2,3),dtype=np.uint8)
    c,_=t.step(rgb,286)
    assert c.tolist()==[286,0,0] and t.delivered_steps==0
    t.observe({'health':100},{'health':80,'finished':False})
    for _ in range(8):t.step(rgb,286)
    assert t.delivered_steps==2000
    assert [(start,n) for start,n,_,active in b.calls if active][-1][1]==284
    assert t.events==1 and b.calls[0][3] is False


def test_overlap_dose_and_terminal_damage_are_explicit():
    b=PulseBrain();t=DamageTraining(b);rgb=np.zeros((2,2,3),dtype=np.uint8)
    t.observe({'health':100},{'health':80,'finished':False});t.step(rgb,1000)
    t.observe({'health':80},{'health':60,'finished':False});t.step(rgb,1000)
    t.observe({'health':60},{'health':-5,'finished':True});t.new_round()
    assert t.events==2 and t.terminal_events==1
    assert t.delivered_steps==2000 and t.cancelled_steps==1000
    t.step(rgb,286);assert t.last_steps==0


def test_frozen_arm_receives_same_stimulus_without_enabling_plasticity():
    b=PulseBrain();t=DamageTraining(b,False)
    t.observe({'health':100},{'health':80,'finished':False})
    t.step(np.zeros((2,2,3),dtype=np.uint8),286)
    assert b.weights_frozen and b.calls==[(0,286,False,True)]
    saved=t.state();other=DamageTraining(b,False);other.restore(saved)
    assert other.state()==saved
    with pytest.raises(ValueError):other.restore({**saved,'until':-1})


def test_full_learning_and_decoder_checkpoint_continues_exactly(tmp_path):
    from test_doom_learning_v6 import brain
    b=brain(tmp_path);controls=NeuralControls([])
    b.step([],100,learning=True,stimulation=([0,2],20),lamina_bias=0)
    game=SimpleNamespace(episode=3,tick=20,observation=lambda:{'episode':3,'tick':20,'health':80})
    store=TrainingCheckpoints(tmp_path/'save',{'model':'test-v6'})
    store.save(b,controls,game,{'training':{'until':b.cursor+1000},'tick':20})
    c,_=b.step([],200,learning=True,stimulation=([0,2],20),lamina_bias=0)
    expected={k:getattr(b,k).copy() for k in ['weight',*b.fields]}
    recovery=store.restore(b,controls,game)
    assert game.episode==4 and game.tick==0 and recovery['interrupted_round']['health']==80
    d,_=b.step([],200,learning=True,stimulation=([0,2],20),lamina_bias=0)
    np.testing.assert_array_equal(c,d)
    for k,v in expected.items():np.testing.assert_array_equal(v,getattr(b,k),err_msg=k)
    with pytest.raises(ValueError,match='another model'):
        TrainingCheckpoints(tmp_path/'save',{'model':'wrong'}).restore(b,controls,game)
    pointer=json.loads((tmp_path/'save/latest.json').read_text())['generation']
    p=tmp_path/'save'/pointer/'brain.npz';p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='checksum'):store.restore(b,controls,game)
