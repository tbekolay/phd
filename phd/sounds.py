import nengo
import nengo.utils.numpy as npext
import numpy as np
from scipy.io.wavfile import read as readwav
from scipy.signal import resample


class FuncProcess(nengo.processes.Process):
    """Psych! Not a process, just a function.

    Implemented so that we can use functions and
    processes interchangeably without having to
    write annoying conditionals.
    """
    def func(self, t):
        raise NotImplementedError()

    def make_step(self, size_in, size_out, dt, rng):
        return self.func


class ArrayProcess(nengo.processes.Process):
    """Psych! Not a process, just going through a vector.

    I guess it's a bit fancier because we can loop
    or stop it after it's done... still though.
    """
    def __init__(self, array, at_end='loop'):
        self.array = array
        # Possible at_end values:
        #   loop: start again from the start
        #   stop: output silence (0) after sound
        assert at_end in ('loop', 'stop')
        self.at_end = at_end
        super(ArrayProcess, self).__init__()

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        assert size_out == (1 if self.array.ndim == 1 else self.array.shape[1])

        rate = 1. / dt

        if self.at_end == 'loop':

            def step_arrayloop(t):
                idx = int(t * rate) % self.array.shape[0]
                return self.array[idx]
            return step_arrayloop

        elif self.at_end == 'stop':

            def step_arraystop(t):
                idx = int(t * rate)
                if idx > self.array.shape[0]:
                    return 0.
                else:
                    return self.array[idx]
            return step_arraystop


class Tone(FuncProcess):
    """A pure tone."""
    def __init__(self, freq_in_hz, rms=0.5):
        self.freq_in_hz = freq_in_hz
        self.rms = rms
        super(Tone, self).__init__()

    @property
    def rms(self):
        return self._rms

    @rms.setter
    def rms(self, _rms):
        self._rms = _rms
        self.amplitude = _rms * np.sqrt(2)

    def func(self, t):
        return self.amplitude * np.sin(2 * np.pi * t * self.freq_in_hz)


class WhiteNoise(nengo.processes.WhiteNoise):
    def __init__(self, rms=0.5):
        # root mean square == standard deviation when mean is 0
        self.rms = rms
        super(WhiteNoise, self).__init__(
            nengo.dists.Gaussian(mean=0, std=rms), scale=False)


class WavFile(nengo.processes.Process):
    def __init__(self, path, at_end='loop', rms=0.5):
        self.path = path
        # Possible at_end values:
        #   loop: start again from the start
        #   stop: output silence (0) after sound
        assert at_end in ('loop', 'stop')
        self.at_end = at_end
        self.rms = rms
        super(WavFile, self).__init__()

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        assert size_out == 1

        rate = 1. / dt

        orig_rate, orig = readwav(self.path)
        new_size = orig.size * (rate / orig_rate)
        wave = resample(orig, new_size)
        wave -= wave.mean()

        # Normalize wave to desired rms)
        wave_rms = npext.rms(wave)
        wave *= (self.rms / wave_rms)

        if self.at_end == 'loop':

            def step_wavfileloop(t):
                idx = int(t * rate) % wave.size
                return wave[idx]
            return step_wavfileloop

        elif self.at_end == 'stop':

            def step_wavfilestop(t):
                idx = int(t * rate)
                if idx > wave.size:
                    return 0.
                else:
                    return wave[idx]
            return step_wavfilestop
