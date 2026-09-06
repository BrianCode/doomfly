"""Reinforcement schedules, without any access to a game's action decoder."""
import numpy as np


def shifted_exposure(schedule, observed_tics, seed):
    """Shift the observed interval, retaining all delivered pulses within it.

    Zero-padding a short lived trial to a longer horizon before shifting can
    hide pulses after the same death time. Avoid that artifact. The recipient
    can still die earlier, so callers must compare actual delivered exposure.
    """
    a=np.asarray(schedule,dtype=bool)
    if a.ndim!=1 or not 0<=observed_tics<=len(a):raise ValueError('Invalid observed interval')
    if a[observed_tics:].any():raise ValueError('Exposure exists outside observed interval')
    result=np.zeros_like(a)
    if observed_tics<2:return a.copy(),0
    rng=np.random.default_rng(seed)
    shift=int(rng.integers(max(1,observed_tics//3),max(2,2*observed_tics//3)))
    result[:observed_tics]=np.roll(a[:observed_tics],shift)
    return result,shift
