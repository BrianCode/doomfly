"""Meaningful causal/numerical boundaries, not a biological validation."""
import json
from pathlib import Path
import numpy as np
import pytest
from doom.engine import NeuralControls
from doom.game import retinal_samples
from doom.reward import SugarReinforcement

def test_pixel_projection_black_white_and_locality():
    uv=np.array([[0,0],[1,1],[.5,.5]],dtype=np.float32)
    image=np.zeros((4,4,3),dtype=np.uint8)
    assert np.all(retinal_samples(image,uv)==0)
    image.fill(255);np.testing.assert_allclose(retinal_samples(image,uv),1,atol=1e-6)
    image.fill(0);image[0,0]=255
    assert retinal_samples(image,uv).tolist()==[1,0,0]

def test_controls_are_neural_and_do_not_auto_fire():
    rows=[{'index':0,'id':'1','type':'DNa02','side':'L'}, {'index':1,'id':'2','type':'DNa02','side':'R'},
      {'index':2,'id':'3','type':'DNp09','side':'L'},{'index':3,'id':'4','type':'MDN','side':'L'},
      {'index':4,'id':'5','type':'MN9','side':'L'}]
    decoder=NeuralControls(rows)
    a=decoder.decode(np.zeros(5),.1)
    assert a['turn']==0 and a['forward']==0 and not a['attack']
    a=decoder.decode(np.array([0,2,3,0,1]),.1)
    assert a['turn']>0 and a['forward']>0 and a['attack']
    a=decoder.decode(np.zeros(5),.1)
    assert not a['attack'] # no fire from lingering filtered activity
    for _ in range(100):a=decoder.decode(np.zeros(5),.1)
    assert abs(a['turn'])<1e-30 and abs(a['forward'])<1e-30

def test_reward_is_stimulus_only_and_expires():
    r=SugarReinforcement(True);r.observe(-1,0);assert not r.active(0)
    r.observe(1,100);assert r.active(299) and not r.active(300)
    assert r.pulses==1
    off=SugarReinforcement(False);off.observe(10,0);assert not off.active(0)

def toy_graph(tmp_path):
    rng=np.random.default_rng(31);n=32
    ptr=np.arange(0,3*n+1,3,dtype=np.int64)
    p=tmp_path/'toy.npz'
    np.savez(p,ptr=ptr,post=rng.integers(0,n,3*n,dtype=np.int32),weight=rng.uniform(-4,5,3*n).astype(np.float32),
      ids=np.arange(n,dtype=np.int64),retina=np.array([0,1],dtype=np.int32),uv=np.array([[0,0],[1,1]],dtype=np.float32),
      lamina=np.array([2,3],dtype=np.int32),sugar=np.array([4],dtype=np.int32),superclass=np.array(['test']*n))
    return p

def test_lazy_kernel_matches_dense_reference_with_changing_inputs(tmp_path):
    from doom.engine import Brain
    from doom.native import NativeBrain
    path=toy_graph(tmp_path);a=Brain(path);b=NativeBrain(path)
    for light,sugar in [([0,0],False),([1,.2],False),([0,0],True),([.4,1],False),([0,0],False)]:
        ca,_=a.step(np.array(light),50,sugar=sugar);cb,_=b.step(np.array(light),50,sugar=sugar)
        np.testing.assert_array_equal(ca,cb)
        np.testing.assert_allclose(a.v,b.v,atol=.002,rtol=0)
        np.testing.assert_allclose(a.g,b.g,atol=.002,rtol=0)

def test_all_edges_disconnected_blocks_downstream_activity(tmp_path):
    from doom.native import NativeBrain
    b=NativeBrain(toy_graph(tmp_path));b.weight.fill(0)
    c,_=b.step(np.array([1.,1.]),300)
    assert c[:4].sum()>0 and c[4:].sum()==0

def test_game_receives_only_decoded_buttons_and_advances():
    from doom.game import Game
    import vizdoom as vzd
    g=Game()
    try:
        tick=g.game.get_episode_time()
        assert g.pixels().shape==(480,640,3)
        g.act({'turn':0.,'forward':0.,'attack':False})
        assert g.game.get_episode_time()==tick+1
        angle=g.game.get_game_variable(vzd.GameVariable.ANGLE)
        g.act({'turn':5.,'forward':0.,'attack':False})
        delta=(g.game.get_game_variable(vzd.GameVariable.ANGLE)-angle+180)%360-180
        assert delta<0 # Positive joystick value really does turn right.
        assert g.game.get_available_buttons()==[vzd.Button.TURN_LEFT_RIGHT_DELTA,vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,vzd.Button.ATTACK]
    finally:g.close()

def test_bci_is_a_fixed_neural_readout_not_a_game_policy():
    rows=[{'index':0,'id':'10059','type':'DNp20','side':'R'},{'index':1,'id':'10162','type':'DNp20','side':'L'},
      {'index':2,'id':'10527','type':'DNpe017','side':'L'},{'index':3,'id':'555871','type':'DNpe017','side':'R'}]
    d=NeuralControls(rows,mode='bci');a=d.decode(np.zeros(4),.1)
    assert a['turn']==a['forward']==0 and not a['attack']
    a=d.decode(np.array([2,0,1,0]),.1)
    assert a['turn']>0 and a['forward']>0 and a['attack']
    a=d.decode(np.zeros(4),.1);assert not a['attack']

def test_native_rejects_invalid_arrays_before_ffi(tmp_path):
    from doom.native import NativeBrain
    good=toy_graph(tmp_path)
    with np.load(good) as f: data={k:f[k] for k in f.files}
    data['post'][0]=1000
    bad=tmp_path/'bad.npz';np.savez(bad,**data)
    with pytest.raises(ValueError,match='out of bounds'):NativeBrain(bad)
    with pytest.raises(ValueError,match='dt=0.1'):NativeBrain(good,dt=.2)
