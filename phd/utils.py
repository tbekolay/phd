import numpy as np
from nengo.utils.compat import is_iterable, is_string


def rescale(val, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((val - old_min) * new_range) / old_range) + new_min


def hz2mel(hz):
    """Convert a value in Hertz to Mels."""
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz."""
    return 700 * (10 ** (mel / 2595.0) - 1)


def hz2st(hz, reference=16.35159783):
    """Convert hertz to semi-tones, relative to musical note C0."""
    if hz < 1.0:
        return 1.0
    return 12 * np.log2(hz / reference)


def st2hz(st, reference=16.35159783):
    """Convert semi-tones to hertz, relative to musical note C0."""
    return reference * np.power(2, st / 12.)


def ensuretuple(val):
    if val is None:
        return val
    if is_string(val) or not is_iterable(val):
        return tuple([val])
    return tuple(val)
