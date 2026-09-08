"""GPU (cuTile) backend: same causal boundaries as the CPU kernels, plus state interop."""
import numpy as np
import pytest
from tests.test_doom import toy_graph

cutile = pytest.importorskip('doom.cutile_brain')
if not cutile.AVAILABLE: pytest.skip('cuTile/CuPy stack unavailable', allow_module_level=True)

def test_cutile_matches_dense_reference_with_changing_inputs(tmp_path):
    from doom.engine import Brain
    path=toy_graph(tmp_path);a=Brain(path);b=cutile.CuTileBrain(path)
    for light,sugar in [([0,0],False),([1,.2],False),([0,0],True),([.4,1],False),([0,0],False)]:
        ca,_=a.step(np.array(light),50,sugar=sugar);cb,_=b.step(np.array(light),50,sugar=sugar)
        np.testing.assert_array_equal(ca,cb)
        np.testing.assert_allclose(a.v,b.v,atol=.002,rtol=0)
        np.testing.assert_allclose(a.g,b.g,atol=.002,rtol=0)
        np.testing.assert_array_equal(a.refractory,b.refractory)

def test_cutile_all_edges_disconnected_blocks_downstream_activity(tmp_path):
    b=cutile.CuTileBrain(toy_graph(tmp_path));b.weight.fill(0);b.mark_weights_changed()
    c,_=b.step(np.array([1.,1.]),300)
    assert c[:4].sum()>0 and c[4:].sum()==0

def test_cutile_rejects_invalid_arrays_before_device_upload(tmp_path):
    p=toy_graph(tmp_path);a=dict(np.load(p));a['post'][0]=1000;np.savez(p,**a)
    with pytest.raises(ValueError,match='out of bounds'):cutile.CuTileBrain(p)

def test_cutile_state_exports_to_cpu_format_and_back(tmp_path):
    """Mid-flight spikes (the delay queue) survive GPU -> CPU -> GPU transfers."""
    from doom.native import NativeBrain
    from doom.checkpoint import FIELDS
    path=toy_graph(tmp_path);g=cutile.CuTileBrain(path);c=NativeBrain(path)
    light=np.array([1.,.6])
    g.step(light,50)
    for _ in range(400):                          # stop at a substep with spikes still in flight
        g.step(light,.1);g.materialize_host_state()
        if g.queue_count.sum()>0:break
    assert g.queue_count.sum()>0, 'expected undelivered spikes in the exported queue'
    for k in FIELDS:getattr(c,k)[:]=getattr(g,k)
    c.cursor=g.cursor;c.sim_ms=g.sim_ms;c.total_spikes=g.total_spikes
    h=cutile.CuTileBrain(path)
    for k in FIELDS:getattr(h,k)[:]=getattr(g,k)
    h.cursor=g.cursor;h.sim_ms=g.sim_ms;h.total_spikes=g.total_spikes;h.load_state_from_host()
    for _ in range(6):
        cg,_=g.step(light,10);cc,_=c.step(light,10);ch,_=h.step(light,10)
        np.testing.assert_array_equal(cg,cc);np.testing.assert_array_equal(cg,ch)
        np.testing.assert_allclose(g.v,c.v,atol=.002,rtol=0);np.testing.assert_allclose(g.g,c.g,atol=.002,rtol=0)
        np.testing.assert_array_equal(g.v,h.v);np.testing.assert_array_equal(g.g,h.g)

def test_cutile_single_substep_calls_match_batched_calls(tmp_path):
    path=toy_graph(tmp_path);a=cutile.CuTileBrain(path);b=cutile.CuTileBrain(path)
    # Dark retina: the per-call luminance filter would otherwise differ by cadence.
    light=np.array([0.,0.]);total=np.zeros(a.n,dtype=np.int32)
    for _ in range(40):total+=a.step(light,.1,lamina_bias=25.)[0]
    cb,_=b.step(light,4.,lamina_bias=25.)
    np.testing.assert_array_equal(total,cb)
    np.testing.assert_allclose(a.v,b.v,atol=1e-4,rtol=0);np.testing.assert_allclose(a.g,b.g,atol=1e-4,rtol=0)


def v6_pair(tmp_path):
    from tests.test_doom_learning_v6 import brain as cpu_brain
    from doom_learning_v6.cutile import CuTileMemoryBrain
    c=cpu_brain(tmp_path)
    g=CuTileMemoryBrain(tmp_path/'graph.npz',eta=.001,circuit=c.circuit,modulation_mask=c.modulation_mask)
    return c,g

def test_cutile_v6_matches_cpu_memory_kernel(tmp_path):
    c,g=v6_pair(tmp_path)
    for stimulus in [([0],20),([2],20),([0,2],20),([0],20)]:
        a,_=c.step([],100,learning=True,stimulation=stimulus,lamina_bias=0)
        d,_=g.step([],100,learning=True,stimulation=stimulus,lamina_bias=0)
        np.testing.assert_array_equal(a,d)
        np.testing.assert_allclose(c.v,g.v,atol=.002,rtol=0)
        np.testing.assert_allclose(c.adaptation,g.adaptation,atol=.002,rtol=0)
        np.testing.assert_allclose(c.weight,g.weight,rtol=1e-6,atol=0)     # device rule: float64 ulp-level differences
        np.testing.assert_allclose(c.memory_w,g.memory_w,rtol=1e-9,atol=1e-15)
    assert g.memory_u[0]<0 and g.weight[0]<20

def test_cutile_v6_multi_bin_call_matches_cpu(tmp_path):
    """A 300 ms call queues three rate-rule bins on the device before one download."""
    c,g=v6_pair(tmp_path)
    for stimulus in [([0],20),([2],20),([0,2],20)]:
        a,_=c.step([],300,learning=True,stimulation=stimulus,lamina_bias=0)
        d,_=g.step([],300,learning=True,stimulation=stimulus,lamina_bias=0)
        np.testing.assert_array_equal(a,d)
        np.testing.assert_allclose(c.v,g.v,atol=.002,rtol=0)
        np.testing.assert_allclose(c.memory_u,g.memory_u,rtol=1e-9,atol=1e-15)
        np.testing.assert_allclose(c.weight,g.weight,rtol=1e-6,atol=0)
    assert g.total_spikes==c.total_spikes and g.cursor==c.cursor

def test_cutile_v6_full_state_checkpoint_reproduces_ongoing_memory(tmp_path):
    _,b=v6_pair(tmp_path)
    b.step([],100,learning=True,stimulation=([0],20),lamina_bias=0)
    b.step([],100,learning=True,stimulation=([2],12),lamina_bias=0)
    assert b.memory_u[0]<0 and b.weight[0]<20
    p=tmp_path/'checkpoint.npz';b.checkpoint(p)
    c,_=b.step([],200,learning=True,stimulation=([0,2],20),lamina_bias=0)
    expected={k:getattr(b,k).copy() for k in ['weight',*b.fields]}
    b.restore(p);d,_=b.step([],200,learning=True,stimulation=([0,2],20),lamina_bias=0)
    np.testing.assert_array_equal(c,d)
    for k,v in expected.items():np.testing.assert_array_equal(v,getattr(b,k),err_msg=k)

def test_cutile_v6_checkpoint_is_loadable_by_the_cpu_brain_after_build_rewrite(tmp_path):
    """Same arrays, same provenance; only the kernel build record differs."""
    import json
    c,g=v6_pair(tmp_path)
    g.step([],100,learning=True,stimulation=([0],20),lamina_bias=0)
    p=tmp_path/'gpu.npz';g.checkpoint(p)
    with np.load(p,allow_pickle=False) as a:
        arrays={k:a[k] for k in a.files if k!='metadata'};m=json.loads(str(a['metadata']))
    m['build']=c.build;q=tmp_path/'cpu.npz';np.savez(q,metadata=json.dumps(m),**arrays)
    c.restore(q)
    x,_=g.step([],100,learning=True,stimulation=([2],12),lamina_bias=0)
    y,_=c.step([],100,learning=True,stimulation=([2],12),lamina_bias=0)
    np.testing.assert_array_equal(x,y);np.testing.assert_allclose(g.v,c.v,atol=.002,rtol=0)

def test_cutile_v6_reset_keeps_only_memory(tmp_path):
    _,b=v6_pair(tmp_path)
    b.step([],100,learning=True,stimulation=([0],20),lamina_bias=0)
    b.memory_u[:]=-.2;b.memory_w[:]=-.1;b.weight[0]=18;b.reset(keep_memory=True)
    assert b.memory_u[0]==-.2 and b.weight[0]==18 and b.cursor==0
    c,_=b.step([],50,learning=False,stimulation=([0],20),lamina_bias=0)
    assert c.sum()>0
    b.reset();assert b.weight[0]==20 and not b.memory_u.any()

def test_cutile_device_drive_equals_host_recipe(tmp_path):
    """The fused device scatter reproduces the host drive bit for bit (baseline with sugar, v6 with tonic and pulses)."""
    b=cutile.CuTileBrain(toy_graph(tmp_path))
    b.step(np.array([.3,.9]),10,sugar=True)
    np.testing.assert_array_equal(b.dev.drive[:b.n].get(),b.drive)
    c,g=v6_pair(tmp_path);g.tonic[1]=9.87;g.tonic[3]=2.5
    g.step([],20,learning=True,stimulation=[([0,2],20),([3],np.array([1.5],dtype=np.float32))],lamina_bias=0)
    np.testing.assert_array_equal(g.dev.drive[:g.n].get(),g.drive)
    g.tonic+=1.0;g.step([],10,learning=False,stimulation=([2],4.),lamina_bias=0)
    np.testing.assert_array_equal(g.dev.drive[:g.n].get(),g.drive);np.testing.assert_array_equal(g.dev.tonic[:g.n].get(),np.asarray(g.tonic))
