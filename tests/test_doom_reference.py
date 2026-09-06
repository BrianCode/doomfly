"""Independent Brian2 oracle for the declared LIF equations and event schedule."""
import numpy as np
import pytest

@pytest.mark.parametrize('backend', ['dense', 'native'])
@pytest.mark.parametrize('cadence_ms', [.1,10.])
def test_kernel_against_brian2_with_refractory_inputs(tmp_path, backend, cadence_ms):
    import brian2 as b2
    from doom.engine import Brain
    from doom.native import NativeBrain
    b2.start_scope(); b2.prefs.codegen.target = 'numpy'; b2.defaultclock.dt = .1*b2.ms
    n = 4
    pre = np.array([0,0,0,1,1,2,3]); post = np.array([0,1,2,2,3,1,0], dtype=np.int32)
    w = np.array([100,35,-20,45,-15,20,-10], dtype=np.float32)
    path = tmp_path/'oracle.npz'
    np.savez(path, ptr=np.r_[0,np.cumsum(np.bincount(pre,minlength=n))].astype(np.int64),
      post=post, weight=w, ids=np.arange(n,dtype=np.int64), retina=np.array([],dtype=np.int32),
      uv=np.empty((0,2),dtype=np.float32), lamina=np.array([0,3],dtype=np.int32),
      sugar=np.array([],dtype=np.int32), superclass=np.array(['test']*n))
    brain = (Brain if backend == 'dense' else NativeBrain)(path)
    neurons = b2.NeuronGroup(n, '''
      dv/dt = (-52*mV - v + drive + g)/(20*ms) : volt (unless refractory)
      dg/dt = -g/(5*ms) : volt (unless refractory)
      drive : volt
      ''', method='exact', threshold='v > -45*mV', reset='v=-52*mV; g=0*mV', refractory=2.2*b2.ms)
    neurons.v = -52*b2.mV
    synapses = b2.Synapses(neurons, neurons, 'w : volt', on_pre='g += w', delay=1.8*b2.ms)
    synapses.connect(i=pre,j=post); synapses.w=w*b2.mV
    spike = b2.SpikeMonitor(neurons)
    state = b2.StateMonitor(neurons, ['v','g'], record=True, when='end')
    network = b2.Network(neurons,synapses,spike,state)
    actual_counts=[]; actual_v=[]; actual_g=[]
    # Strong autapse arrives within the source's refractory period. Excitatory
    # and inhibitory convergence, no drive, and changed drive cover scheduling.
    for drive, milliseconds in [(12.,40),(0.,10),(18.,40)]:
        neurons.drive = np.array([drive,0,0,drive])*b2.mV
        network.run(milliseconds*b2.ms)
        for _ in range(round(milliseconds/cadence_ms)):
            c,_=brain.step(np.empty(0),cadence_ms,lamina_bias=drive)
            actual_counts.append(c); actual_v.append(brain.v.copy()); actual_g.append(brain.g.copy())
    expected_counts = np.zeros((900,n),dtype=int)
    for i,t in zip(spike.i,spike.t/b2.ms): expected_counts[round(float(t)/.1),i]+=1
    stride=round(cadence_ms/.1)
    np.testing.assert_array_equal(np.asarray(actual_counts),expected_counts.reshape(-1,stride,n).sum(axis=1))
    np.testing.assert_allclose(np.asarray(actual_v).T,(state.v/b2.mV)[:,stride-1::stride],atol=.002,rtol=0)
    np.testing.assert_allclose(np.asarray(actual_g).T,(state.g/b2.mV)[:,stride-1::stride],atol=.002,rtol=0)
