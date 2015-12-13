import brian.hears as bh
import numpy as np

from .utils import hz2mel, mel2hz

# NB! Although the dummy sound is never used, it must be first set
# because Brian Hears isn't really designed for online sounds, which
# NengoSound is. So, we set this then immediately swap it.
dummy_sound = bh.Sound(np.zeros(1))


def erbspace(low, high, n_freq):
    """Sample ERB distribution; low and high in Hz."""
    f = np.linspace(low, high, n_freq) * 0.001  # original f in kHz
    return 6.23 * np.square(f) + 93.39 * f + 28.52


def melspace(low, high, n_freq):
    return mel2hz(np.linspace(hz2mel(low), hz2mel(high), n_freq))


def rectify(filterbank, scale=3):
    """Half wave rectify and scale."""

    def _bm2ihc(x, scale=scale):
        return scale * np.clip(x, 0, np.inf)
    ihc = bh.FunctionFilterbank(filterbank, _bm2ihc)
    ihc.cached_buffer_end = 0  # Fails if we don't do this...
    return ihc


def compress(filterbank, scale=3):
    """Half wave rectify and compress with a 1/3 power law."""

    def _bm2ihc(x, scale=scale):
        return scale * np.clip(x, 0, np.inf) ** (1. / 3.)

    ihc = bh.FunctionFilterbank(filterbank, _bm2ihc)
    ihc.cached_buffer_end = 0  # Fails if we don't do this...
    return ihc


def gammatone(source, freqs, dt, b=1.019):
    duration = int(dt / source.source.dt)
    fb = bh.Gammatone(dummy_sound, freqs, b=b)
    fb.source = source
    fb.buffersize = duration
    return compress(fb, scale=3)


def log_gammachirp(source, freqs, dt, glide_slope=-2.96, time_const=1.81):
    duration = int(dt / source.source.dt)
    fb = bh.LogGammachirp(dummy_sound, freqs, c=glide_slope, b=time_const)
    fb.source = source
    fb.buffersize = duration
    return compress(fb, scale=1.76)


def dual_resonance(source, freqs, dt):
    duration = int(dt / source.source.dt)
    fb = bh.DRNL(dummy_sound, freqs, type='human')
    fb.source = source
    fb.buffersize = duration
    return compress(fb, scale=0.75)


def compressive_gammachirp(source, freqs, dt, update_interval=1):
    duration = int(dt / source.source.dt)
    fb = bh.DCGC(dummy_sound, freqs, update_interval=update_interval)
    fb.source = source
    fb.buffersize = duration
    return compress(fb, scale=0.7)


def tan_carney(source, freqs, dt, update_interval=1):
    duration = int(dt / source.source.dt)
    fb = bh.TanCarney(source, freqs, update_interval=update_interval)
    fb.source = source
    fb.buffersize = duration
    return rectify(fb, scale=7)
