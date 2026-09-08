"""Independent physiological targets; no Doom performance is involved."""
from .visual import VisualMemoryBrain


def calibrated_brain(eta=.001,backend='native'):
    if backend=='cutile':
        from .cutile import CuTileVisualMemoryBrain
        b=CuTileVisualMemoryBrain(eta=eta)
    elif backend=='native':b=VisualMemoryBrain(eta=eta)
    else:raise ValueError('Unknown brain backend')
    b.tonic[b.circuit['mb']]=9.87
    # Best *observed* point in the recorded current sweep, not a bisection
    # interpolation: this recurrent spiking system is not monotonic in bias.
    b.tonic[b.circuit['dan']]=11.3125
    b.dan_baseline_hz[:]=20.09
    b.calibration={'MBON_current':9.87,'DAN_current':11.3125,
        'observed_MBON_hz':[37,37],'observed_DAN_hz':[23,19],
        'target_MBON_hz':37.16625,'target_MBON_sd':9.06014,
        'target_DAN_hz':20.09,'target_DAN_sd':4.30193,
        'evidence':'https://doi.org/10.1038/s41586-024-07819-w',
        'source_data':'Figure 1, panels d/e, n=20 flies per type',
        'limits':'A fitted background current is not identified pacemaker physiology. This only constrains baseline rates, not cue responses, burst statistics or learned behavior.'}
    return b
