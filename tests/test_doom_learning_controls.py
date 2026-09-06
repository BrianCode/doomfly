import numpy as np
import pytest
from doom_learning.controls import shifted_exposure


def test_shuffled_exposure_stays_inside_recorded_alive_interval():
    a=np.zeros(420,dtype=bool);a[[4,50,350,383]]=True
    shifted,offset=shifted_exposure(a,384,17)
    assert offset>0 and shifted.sum()==a.sum()
    assert shifted[:384].sum()==a.sum() and not shifted[384:].any()
    assert not np.array_equal(a,shifted)
    np.testing.assert_array_equal(shifted,shifted_exposure(a,384,17)[0])


def test_invalid_exposure_is_not_silently_discarded():
    with pytest.raises(ValueError):shifted_exposure([False,False,True],2,1)


def test_zero_duration_exposure_remains_empty():
    a=np.zeros(420,dtype=bool)
    shifted,offset=shifted_exposure(a,0,17)
    np.testing.assert_array_equal(a,shifted);assert offset==0
