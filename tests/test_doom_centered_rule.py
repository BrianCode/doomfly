"""Independent mathematical controls for the candidate centered rate rule."""
import numpy as np
import pytest
from doom_learning_v5.rule import advance,PARAMETERS


def state():return [np.zeros(1,dtype=np.float64) for _ in range(4)]


def protocol(kc_first,step=.01):
    s=state();g=np.ones((1,1))
    for tick in range(round(2/step)):
        t=tick*step
        k=10. if (0<=t<.5 if kc_first else .5<=t<1) else 0.
        d=30. if (.5<=t<1 if kc_first else 0<=t<.5) else 0.
        advance(*s,np.array([k]),np.array([d]),g,step,.001)
    return s


def test_forward_and_backward_have_opposite_signs():
    forward=protocol(True);backward=protocol(False)
    assert forward[-1][0]<0 and backward[-1][0]>0
    assert forward[-1][0]==pytest.approx(-backward[-1][0],abs=1e-12)


def test_rate_step_convergence():
    a=protocol(True,.01)[-1][0];b=protocol(True,.005)[-1][0];c=protocol(True,.001)[-1][0]
    assert abs(a-c)<.001*abs(c)
    assert abs(b-c)<abs(a-c)


def test_tonic_baseline_has_no_associative_drive():
    s=state()
    # Actual DAN rate equals its declared baseline; centered input is zero.
    for _ in range(100):advance(*s,np.array([10.]),np.array([0.]),np.ones((1,1)),.01,.001)
    assert s[2][0]==0 and s[3][0]==0


def test_passive_memory_decay_and_frozen_weights():
    s=state();s[2][:]=.2;s[3][:]=.2
    advance(*s,np.array([0.]),np.array([0.]),np.ones((1,1)),.01,.001,learning=False)
    assert s[2][0]==pytest.approx(.2*np.exp(-.01/PARAMETERS['memory_decay_seconds']))
    assert s[3][0]<.2
    old=[x.copy() for x in s]
    advance(*s,np.array([10.]),np.array([50.]),np.ones((1,1)),.01,.001,learning=True,frozen=True)
    np.testing.assert_array_equal(s[2],old[2]);np.testing.assert_array_equal(s[3],old[3])


def test_unrelated_compartment_is_unchanged():
    yk=np.zeros(2);yd=np.zeros(2);u=np.zeros(2);w=np.zeros(2)
    for t in range(100):advance(yk,yd,u,w,np.array([10.,0.]),np.array([0.,30.]),np.eye(2),.01,.001)
    np.testing.assert_array_equal(u,[0.,0.]);np.testing.assert_array_equal(w,[0.,0.])
