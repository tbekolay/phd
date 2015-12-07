import brian as br
import brian.hears as bh
import numpy as np

from .utils import hz2mel, mel2hz

dummy_sound = bh.Sound(np.zeros(1))


def erbspace(low, high, n_freq):
    """Sample ERB distribution; low and high in Hz."""
    return bh.erbspace(low * br.Hz, high * br.Hz, n_freq)


def melspace(low, high, n_freq):
    return mel2hz(np.linspace(hz2mel(low), hz2mel(high), n_freq))


def gammatone(freqs, b=1.019):
    return bh.Gammatone(dummy_sound, freqs, b=b)


def log_gammachirp(freqs, glide_slope=-2.96, time_const=1.81):
    return bh.LogGammachirp(dummy_sound, freqs, c=glide_slope, b=time_const)


# def linear_gammachirp(freqs, glide_slope=0.0, time_const=None):
#     if time_const is None:
#         time_const = np.linspace(3, 0.3, freqs.size) * br.ms
#     return bh.LinearGammachirp(
#         dummy_sound, freqs, time_constant=time_const, c=glide_slope)

# Use LinearGaborChirp instead


def dual_resonance(freqs):
    return bh.DRNL(dummy_sound, freqs, type='human')


def compressive_gammachirp(freqs, update_interval=1):
    return bh.DCGC(dummy_sound, freqs, update_interval=update_interval)


def tan_carney(freqs, update_interval=1):
    return bh.TanCarney(dummy_sound, freqs, update_interval=update_interval)
