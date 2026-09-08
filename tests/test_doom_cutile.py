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
