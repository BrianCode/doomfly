"""Exact process-recovery checks; no claim about biological memory."""
import hashlib
import json
import numpy as np
import pytest
from doom.native import NativeBrain
from doom.engine import NeuralControls
from doom.game import Game,retinal_samples
from doom.checkpoint import Checkpoints
from test_doom import toy_graph


def controls():
    return NeuralControls([{'index':0,'id':'1','type':'DNp20','side':'R'},
      {'index':1,'id':'2','type':'DNpe017','side':'R'}],mode='bci')


def advance(brain,decoder,game,n=10):
    rows=[]
    for _ in range(n):
        pixels=game.pixels();light=retinal_samples(pixels,brain.uv)
        c,_=brain.step(light,28.6);a=decoder.decode(c,.0286);game.act(a)
        rows.append((hashlib.sha256(pixels.tobytes()).hexdigest(),c.tolist(),a,game.observation()))
    return rows


def test_resume_reproduces_neural_state_and_declares_new_round(tmp_path):
    graph=toy_graph(tmp_path);brain=NativeBrain(graph);decoder=controls();game=Game(scenario='combat_survival')
    store=Checkpoints(tmp_path/'recovery',{'model':'toy-technical-test','graph':'fixture'})
    try:
        advance(brain,decoder,game)
        before=game.observation();engine_tick=game.game.get_episode_time();store.save(brain,decoder,game,{'run':'test','tick':10})
        assert game.observation()==before
        assert game.game.get_episode_time()==engine_tick
        expected=[]
        for light in [np.array([0.,1.]),np.array([.4,.7]),np.array([1.,0.])]:
            c,_=brain.step(light,28.6);expected.append((c.tolist(),decoder.decode(c,.0286)))
    finally:game.close()
    restored=NativeBrain(graph);readout=controls();other=Game(scenario='combat_survival')
    try:
        recovery=store.restore(restored,readout,other)
        assert recovery['run']=='test' and recovery['recovery']=='neural-state-with-new-arena'
        assert other.observation()['episode']==2 and other.observation()['tick']==0
        assert recovery['interrupted_round']==before
        actual=[]
        for light in [np.array([0.,1.]),np.array([.4,.7]),np.array([1.,0.])]:
            c,_=restored.step(light,28.6);actual.append((c.tolist(),readout.decode(c,.0286)))
        assert actual==expected
        with pytest.raises(ValueError,match='another model'):
            Checkpoints(tmp_path/'recovery',{'model':'other'}).restore(restored,readout,other)
        generation=json.loads((tmp_path/'recovery/latest.json').read_text())['generation']
        p=tmp_path/'recovery'/generation/'brain.npz';p.write_bytes(p.read_bytes()+b'corruption')
        with pytest.raises(ValueError,match='checksum'):store.restore(restored,readout,other)
    finally:other.close()
